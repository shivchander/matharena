"""This module provides a unified API for querying various large language models."""

import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import anthropic
import requests
from anthropic.types import TextBlock, ThinkingBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from loguru import logger
from openai import OpenAI, RateLimitError
from together import Together
from tqdm import tqdm
from transformers import AutoTokenizer

from matharena.request_logger import request_logger
from matharena.utils import check_for_extra_keys

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None


class APIClient:
    """A client that queries various LLM APIs."""

    def __init__(
        self,
        model,
        timeout=18000,
        max_tokens=None,
        api="openai",
        max_retries=10,
        max_retries_inner=5,
        concurrent_requests=30,
        no_system_messages=False,
        context_limit=None,
        background=False,
        read_cost=1,
        write_cost=1,
        sleep_on_error=60,
        sleep_after_request=0.1,
        include_max_tool_calls=True,
        throw_error_on_failure=False,
        max_tokens_param="max_tokens",
        reasoning_effort=None,
        batch_processing=False,
        use_openai_responses_api=False,
        use_gdm_tools=False,
        max_tool_calls=0,
        tools=None,
        **kwargs,
    ):
        """Initializes the APIClient object and params. All prompts are set at run_queries invocation.

        Args:
            model (str): The name of the model to use.
            timeout (int, optional): The timeout for API requests in seconds. Defaults to 9000.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            api (str, optional): The API to use. Defaults to 'openai'.
            max_retries (int, optional): The maximum number of retries for a failed query. Defaults to 50.
            concurrent_requests (int, optional): The number of concurrent requests to make. Defaults to 30.
            no_system_messages (bool, optional): Whether to disable system messages. Defaults to False.
            read_cost (int, optional): The cost of reading a token. Defaults to 1.
            write_cost (int, optional): The cost of writing a token. Defaults to 1.
            sleep_on_error (int, optional): The number of seconds to sleep on an error. Defaults to 60.
            sleep_after_request (float, optional): The number of seconds to sleep after a request. Defaults to 0.1.
            throw_error_on_failure (bool, optional): Whether to throw an error on failure. Defaults to False.
            max_tokens_param (str, optional): The name of the max_tokens parameter for the API. Defaults to "max_tokens".
            reasoning_effort (str, optional): The reasoning effort to use. Defaults to None.
            batch_processing (bool, optional): Whether to use batch processing. Defaults to False.
            use_openai_responses_api (bool, optional): Whether to use OpenAI responses. Defaults to False.
            max_tool_calls (int|dict, optional): The maximum number of tool calls to make. Defaults to 0.
                Could also be a dict that specifies max calls per tool name.
            tools (list, optional): A list of tools to use. Defaults to None.
            **kwargs: Additional keyword arguments for the API.
        """
        # Max tool calls
        self.tool_calls_allowed = False
        if isinstance(max_tool_calls, int):
            self.max_tool_calls = {"any": max_tool_calls}
            self.max_tool_calls_mode = "total"
            if max_tool_calls > 0:
                self.tool_calls_allowed = True
        elif isinstance(max_tool_calls, dict):
            self.max_tool_calls = max_tool_calls
            self.max_tool_calls_mode = "per_tool"
            if sum(max_tool_calls.values()) > 0:
                self.tool_calls_allowed = True

        # Adapt model name and other args to the model
        if "--" in model:
            model, reasoning_effort = model.split("--")
            logger.info(f"Model: {model}, Reasoning effort: {reasoning_effort}")
        if api not in ["anthropic", "openai"] and batch_processing:
            logger.warning("Batch processing is only supported for the Anthropic API and OpenAI API.")
            batch_processing = False
        if ("o1" in model or "o3" in model or "o4" in model or "gpt-5" in model) and api == "openai":
            logger.info("Not using system messages for o1/o3/o4 model.")
            no_system_messages = True  # o1 model cannot handle system messages
            if not use_openai_responses_api:
                max_tokens_param = "max_completion_tokens"
        if use_openai_responses_api and not batch_processing:
            max_tokens_param = "max_output_tokens"
        if self.tool_calls_allowed and not use_openai_responses_api:
            max_tokens_param = "max_completion_tokens"
        self._kwarg_remover(api, model, kwargs)

        self.model = model
        self.kwargs = kwargs
        self.max_tokens_param = max_tokens_param
        self.context_limit = context_limit
        if max_tokens is not None:
            self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retries_inner = max_retries_inner
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.write_cost = write_cost
        self.background = background
        self.batch_processing = batch_processing
        self.use_openai_responses_api = use_openai_responses_api
        self.include_max_tool_calls = include_max_tool_calls
        self.background = background
        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            if not self.use_openai_responses_api or self.batch_processing:
                self.kwargs["reasoning_effort"] = reasoning_effort
            elif "reasoning" in self.kwargs:
                self.kwargs["reasoning"]["effort"] = reasoning_effort
            else:
                self.kwargs["reasoning"] = {"effort": reasoning_effort}

        # Save tools: user should forward all (even if mix of competition-given and scaffold-given)
        self.tools = tools if tools is not None else []
        self.tool_functions = {
            tool_desc["function"]["name"]: func for func, tool_desc in self.tools if "function" in tool_desc
        }
        self.tool_descriptions = [tool_desc for _, tool_desc in self.tools]
        if (not self.tool_calls_allowed or len(self.tool_descriptions) == 0) and "tool_choice" in self.kwargs:
            del self.kwargs["tool_choice"]

        # Prep api
        self.api = api
        self.api_key = None
        self.base_url = None
        self.terminated = False
        self._initialize_api_keys()

        # VLLM-specific initialization
        if self.api == "vllm":
            if LLM is None:
                raise ImportError("vllm is not installed. pip install vllm")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            vllm_args = {}

            for p in ("temperature", "top_p", "max_tokens"):
                if p in self.kwargs:
                    vllm_args[p] = self.kwargs.pop(p)
            self.sampling_params = SamplingParams(**vllm_args)
            self.vllm_model = LLM(
                model=self.model, tensor_parallel_size=4 if "n_gpus" not in kwargs else kwargs["n_gpus"]
            )
            logger.info(f"Loaded local vllm model `{self.model}` with sampling {kwargs}")

    def terminate(self):
        """Terminates the APIClient."""
        self.terminated = True

    def _kwarg_remover(self, api, model, kwargs):
        """Removes kwargs that are not supported by the API or model.

        Args:
            api (str): The API to use.
            model (str): The model to use.
            kwargs (dict): The kwargs to clean.
        """
        if "use_openai_responses_api_tools" in kwargs:
            del kwargs["use_openai_responses_api_tools"]
        if any([kw in model for kw in ["o1", "o3", "o4"]]) and "temperature" in kwargs:
            del kwargs["temperature"]
        for kwarg in ["top_p", "top_k", "temperature"]:
            if kwarg in kwargs and kwargs[kwarg] is None:
                del kwargs[kwarg]
        if (api == "anthropic" and "claude-3-7" in model) or (("o1" in model or "o3" in model) and api == "openai"):
            for kwarg_to_remove in ["top_p", "top_k", "temperature"]:
                if kwarg_to_remove in kwargs:
                    logger.info(f"Removing {kwarg_to_remove} parameter for {model} model.")
                    del kwargs[kwarg_to_remove]

    def _initialize_api_keys(self):
        """Initializes the API keys and base URLs for the selected API."""
        if self.api == "sri":
            self.api_key = os.getenv("SRI_API_KEY")
            self.base_url = "https://srlx.inf.ethz.ch/openai"
            self.api = "openai"
        elif self.api == "xai":
            self.api_key = os.getenv("XAI_API_KEY")
            self.base_url = "https://api.x.ai/v1"
            self.api = "openai"
        elif self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            """
                NOTE: We generally use google through openai compat
                with external tools. BCN should be tested with internal
                ones and this doesn't seem to be possible via compat,
                so we route to google via requests.
            """
            if self.tool_calls_allowed and "gdm-eval-model-bcn" in self.model:
                self.api = "google"
                self.base_url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/gdm-eval-model-bcn:generateContent"
                )
            else:
                self.api = "openai"  # !
                self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            if self.tool_calls_allowed:
                self.api = "openai"
                self.base_url = "https://api.anthropic.com/v1/"
                if "thinking" in self.kwargs:
                    self.kwargs["extra_body"] = self.kwargs["thinking"]
                    del self.kwargs["thinking"]
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "glm":
            self.api_key = os.getenv("GLM_API_KEY")
            self.base_url = "https://api.z.ai/api/paas/v4/"
            self.api = "openai"
        elif self.api == "tiiuae":
            self.api_key = os.getenv("TIIUAE_API_KEY")
            self.base_url = "https://falcon-stage.blueoc.tech/v1"
            self.api = "openai"
        elif self.api == "moonshot":
            self.api_key = os.getenv("MOONSHOT_API_KEY")
            self.base_url = "https://api.moonshot.ai/v1"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "deepseek_special":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com/v3.2_speciale_expires_on_20251215"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            if "via_openai" in self.kwargs:
                del self.kwargs["via_openai"]
                self.api = "openai"
        elif self.api == "vllm":
            return
        else:
            raise ValueError(f"API {self.api} not supported.")

        assert self.api_key is not None, "API key not found."

    class InternalRequestResult:
        """A class to hold the result of a request internally (below run_queries)."""

        def __init__(self, conversation, input_tokens, output_tokens, n_retries=0, time=0, reasoning_tokens=0):
            self.conversation = conversation
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.n_retries = n_retries
            self.time = time
            self.reasoning_tokens = reasoning_tokens

    def run_queries(self, queries, no_tqdm=False, ignore_tool_calls=False, custom_indices=None):
        """Only entry point: runs a given list of queries through the API.

        Args:
            queries (list[MessageList]): A list of queries to run. Each query is a MessageList that we will mangle into
            the right format for this API.
            no_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction. Defaults to False.

        Yields:
            tuple: An (idx, conversation, detailed_cost) tuple.
                idx: Integer index of the query this response corresponds to in [0, len(queries)-1].
                conversation: Full list of messages (including those from the query) in the API format (incl. CoT).
                detailed_cost: A dict with total "cost" ($), "input_tokens", "output_tokens", and "time" (seconds).
        """
        if not no_tqdm:
            logger.info(f"Running {len(queries)} queries.")

        # For now only switches between system/developer, keeps rest intact
        queries = [self._validate_and_prepare_query(query) for query in queries]

        # Prepare batch indices, agents use custom ones for request_logger
        if custom_indices is not None:
            indices = custom_indices
        else:
            indices = list(range(len(queries)))

        # Case 1: VLLM
        if self.api == "vllm":
            # Bypass threading and batch everything into one local generate
            # TODO indices and request logger
            yield from self._run_vllm_queries(queries)
            return

        # Case 2: Batch API
        if self.batch_processing:
            start_time = time.time()
            if self.api == "openai":
                results_batch = self._openai_batch_processing(queries, indices)
            else:
                results_batch = self._anthropic_batch_processing(queries, indices)
            end_time = time.time()  # pack all time into first index (since batch)
            for idx, result in enumerate(results_batch):
                if result is None:
                    conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": ""}]
                    result = self.InternalRequestResult(conversation, input_tokens=0, output_tokens=0)
                detailed_cost = {
                    "cost": self._get_cost(result.input_tokens, result.output_tokens),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "reasoning_tokens": result.reasoning_tokens,
                    "time": end_time - start_time,
                    "retries": result.n_retries,
                }
                yield idx, result.conversation, detailed_cost
            return

        # Case 3: Standard API; parallelize manually
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            future_to_index = {
                executor.submit(self._run_query_with_retry, idx, query, ignore_tool_calls): idx
                for idx, query in zip(indices, queries)
            }

            iterator = as_completed(future_to_index)
            if not no_tqdm:
                iterator = tqdm(iterator, total=len(future_to_index))

            for future in iterator:
                idx = future_to_index[future]
                result = future.result()
                if result is None:
                    conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": ""}]
                    result = self.InternalRequestResult(conversation, input_tokens=0, output_tokens=0)
                detailed_cost = {
                    "cost": self._get_cost(result.input_tokens, result.output_tokens),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "reasoning_tokens": result.reasoning_tokens,
                    "n_retries": result.n_retries,
                    "time": time.time() - start_time,
                    "request_time": result.time,
                }
                yield idx, result.conversation, detailed_cost

    def _validate_and_prepare_query(self, query):
        """Prepares a query for the API.
            All "tool_response" and "assistant" blocks must have come straight from this APIClient
                => We only need to normalize the developer and user messages
            We will assume they arrive in normalized format (only role: "user"/"developer" and content:str fields)

        Args:
            query (MessageList): List of messages to prepare.

        Returns:
            query_prepared (MessageList): The prepared conversation in the format for this API.
        """
        query_prepared = []
        for m in query:
            query_prepared.append(m.copy())
            if m.get("role", "") == "developer" and not self.no_system_messages:
                query_prepared[-1]["role"] = "system"  # use system if expected by API
            if m.get("role", "") == "user":
                check_for_extra_keys(m, ["role", "content"])

            # Fix images into another format for gemini and grok and qwen and glm
            if (
                m.get("role", "") == "user"
                and isinstance(m.get("content", ""), list)
                and (
                    "gemini-" in self.model
                    or "riftrunner" in self.model
                    or "gdm-eval" in self.model
                    or "grok-4" in self.model
                    or "qwen" in self.model
                    or "glm" in self.model
                )
            ):
                new_content = []
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type", "") == "input_image" and "image_url" in block:
                        inner = {"url": block["image_url"]}
                        if "grok-4" in self.model:
                            inner["detail"] = "high"
                        new_content.append({"type": "image_url", "image_url": inner})
                    elif isinstance(block, dict) and block.get("type", "") == "input_text" and "text" in block:
                        new_content.append({"type": "text", "text": block["text"]})
                    elif isinstance(block, dict) and block.get("type", "") in ["text", "image_url"]:
                        # Already transformed (e.g., during last_chance), keep as-is
                        new_content.append(block)
                    else:
                        # Unknown block type, log warning but keep the block to avoid data loss
                        logger.warning(f"Unknown content block type in user message: {block}")
                        new_content.append(block)
                query_prepared[-1]["content"] = new_content

            # Fix images for anthropic
            if m.get("role", "") == "user" and isinstance(m.get("content", ""), list) and self.api == "anthropic":
                new_content = []
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type", "") == "input_image" and "image_url" in block:
                        b64_full = block["image_url"]
                        tag = "data:image/png;base64,"
                        assert b64_full.startswith(tag)
                        source = {"type": "base64", "media_type": "image/png", "data": b64_full[len(tag) :]}
                        new_content.append({"type": "image", "source": source})
                    elif isinstance(block, dict) and block.get("type", "") == "input_text" and "text" in block:
                        new_content.append({"type": "text", "text": block["text"]})
                    elif isinstance(block, dict) and block.get("type", "") in ["text", "image"]:
                        # Already transformed (e.g., during last_chance), keep as-is
                        new_content.append(block)
                    else:
                        # Unknown block type, log warning but keep the block to avoid data loss
                        logger.warning(f"Unknown content block type in user message: {block}")
                        new_content.append(block)
                query_prepared[-1]["content"] = new_content
        return query_prepared

    def _get_cost(self, input_tokens, output_tokens):
        return (input_tokens * self.read_cost + output_tokens * self.write_cost) / 1e6

    def _get_messages_from_anthropic_content(self, content):
        """Postprocesses the content from an Anthropic API query.

        Args:
            content: The content from the Anthropic API.

        Returns:
            str: The textual representation.
        """
        messages = []
        for content_block in content:
            if isinstance(content_block, ThinkingBlock):
                messages.append({"role": "assistant", "type": "reasoning", "content": content_block.thinking})
            elif isinstance(content_block, TextBlock):
                messages.append({"role": "assistant", "type": "response", "content": content_block.text})
                break
        return messages

    def _drop_cot(self, messages):
        """Drops all CoT/thinking/reasoning messages from a conversation.
        This is a cost saving measure at API call site; conversations that are maintained will have this.

        Args:
            messages (MessageList): The conversation to drop CoT from.
        Returns:
            MessageList: The conversation without CoT messages.
        """
        new_messages = []
        for m in messages:
            if m.get("role", "") == "assistant" and m.get("type", "response") == "cot":
                continue
            new_messages.append(m)

            # Remove thought signatures for riftrunner and bcn
            if m.get("thought_signature", None) is not None:
                new_messages[-1].pop("thought_signature")

            # Remove thought signatures for riftrunner and bcn
            if m.get("extra_content", None) is not None:
                if m["extra_content"].get("google", None) is not None:
                    if m["extra_content"]["google"].get("thought_signature", None) is not None:
                        new_messages[-1]["extra_content"]["google"].pop("thought_signature")

        return new_messages

    """
        Case 1: VLLM
    """

    def _run_vllm_queries(self, queries):
        """
        Batch all queries into one vllm.generate(...) call, collect final states, yield in order.

        Args:
            queries (List[str]): A list of queries (chat formatted convos) to run.

        Yields:
            tuple: An (idx, conversation, detailed_cost) tuple.
        """
        tasks = []
        for idx, query in enumerate(queries):
            query_in_template = self.tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
            tasks.append({"id": str(idx), "prompt": query_in_template})

        logger.info(f"Running {len(tasks)} queries on local vllm…")
        last_outputs = []
        time_start = time.time()
        for batch in self.vllm_model.generate(tasks, sampling_params=self.sampling_params):
            for out in batch.outputs:
                last_outputs.append(out)
            if self.terminated:
                last_outputs += [None for _ in range(len(tasks) - len(last_outputs))]
        time_end = time.time()

        for idx, out in enumerate(last_outputs):
            if out is None:
                text = ""
            else:
                text = out.text
            conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": text}]
            inp = getattr(out, "n_input_tokens", 0)
            outp = getattr(out, "n_output_tokens", 0)
            detailed_cost = {
                "cost": self._get_cost(inp, outp),
                "input_tokens": inp,
                "output_tokens": outp,
                "retries": 0,
                "time": time_end - time_start if idx == 0 else 0,  # put all in first since batch
            }
            yield idx, conversation, detailed_cost

    """
        Case 2: Batch API
    """

    def _openai_batch_processing(self, queries, indices, retry_idx=0):
        """Processes a batch of queries  using the OpenAI API.

        Args:
            queries (list[MessageList]): A list of queries to run, each is a MessageList in API format.
            retry_idx (int, optional): Current retry index starting from 0.

        Returns:
            list: A list of InternalRequestResult or None.
        """
        if retry_idx >= self.max_retries:
            return [None for _ in range(len(queries))]

        jsonl_queries = []
        for idx, query in zip(indices, queries):
            request = {
                "custom_id": f"apiquery-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": self.model, "messages": self._drop_cot(query), **self.kwargs},
            }
            jsonl_queries.append(request)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

        # create temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        with open(tmp.name, "wb") as f:
            for i, query in enumerate(jsonl_queries):
                f.write(json.dumps(query).encode("utf-8"))
                f.write(b"\n")

        batch_input_file = client.files.create(file=open(tmp.name, "rb"), purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # close tmp file
        tmp.close()

        logger.info(
            f"Running {len(queries)} queries with batch ID {batch.id} using file with File ID {batch_input_file.id}."
        )

        request_counts = dict(batch.request_counts)

        while True:
            try:
                batch = client.batches.retrieve(batch.id)
            except Exception as e:  # noqa: E722 F841
                logger.warning(f"Error connecting to batch OpenAI. Retrying in 10s. Exception: {e}")
                pass
            if any([request_counts[key] != dict(batch.request_counts)[key] for key in request_counts]):
                request_counts = dict(batch.request_counts)
                logger.info(
                    f"Completed Requests Progress: {request_counts['completed']}/{len(queries)}. Errors: {request_counts['failed']}/{len(queries)}"
                )
            if batch.status == "completed":
                break
            time.sleep(10)

        results = [None for _ in range(len(queries))]
        repeat_indices = []

        if batch.output_file_id is None:
            return results
        while True:
            try:
                file_response = client.files.content(file_id=batch.output_file_id)
                break
            except Exception as e:
                logger.error(f"Error connecting to batch OpenAI. Retrying in 10s. Exception: {e}")
                time.sleep(10)
                continue

        json_response = []
        for line in file_response.iter_lines():
            json_response.append(json.loads(line))

        for result in json_response:
            index = int(result["custom_id"].split("-")[-1])
            if result["response"]["status_code"] != 200:
                repeat_indices.append(index)
                logger.error(f"Error: {result['response']['status_code']}")
            else:
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    conversation = [m.copy() for m in queries[index]] + [{"role": "assistant", "content": content}]
                    input_tokens = result["response"]["body"]["usage"]["prompt_tokens"]
                    output_tokens = result["response"]["body"]["usage"]["completion_tokens"]
                    results[index] = self.InternalRequestResult(conversation, input_tokens, output_tokens, n_retries=retry_idx)
                except Exception as e:
                    logger.error(f"Error when unpacking batch OpenAI response, will repeat. Exception: {e}")
                    repeat_indices.append(index)

        for i in range(len(results)):
            if results[i] is None:
                repeat_indices.append(i)
        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_results = self._openai_batch_processing(repeat_queries, retry_idx + 1)
            for i, result in zip(repeat_indices, repeat_results):
                results[i] = result

        return results

    def _anthropic_batch_processing(self, queries, indices, retry_idx=0):
        """Processes a batch of queries using the Anthropic API.

        Args:
            queries (list[MessageList]): A list of queries to run, each is a MessageList in API format.
            retry_idx (int, optional): Current retry index starting from 0.

        Returns:
            list: A list of InternalRequestResult or None.
        """
        if retry_idx >= self.max_retries:
            return [None for _ in range(len(queries))]

        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
        )

        requests = []
        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{datetime.now().microsecond:06d}"
        for idx, query in zip(indices, queries):
            kwargs_here = self.kwargs.copy()
            if query[0]["role"] == "system":
                kwargs_here["system"] = query[0]["content"]
                query = query[1:]

            payload = {
                "custom_id": f"apiquery-{idx}",
                "params": {"model": self.model, "messages": self._drop_cot(query), **kwargs_here},
            }
            request_logger.log_request(ts=ts, batch_idx=idx, request=payload)
            request = Request(
                custom_id=f"apiquery-{idx}",
                params=MessageCreateParamsNonStreaming(model=self.model, messages=self._drop_cot(query), **kwargs_here),
            )
            requests.append(request)

        message_batch = client.messages.batches.create(requests=requests)

        logger.info(f"Running {len(queries)} queries with batch ID {message_batch.id}")

        current_request_counts = dict(message_batch.request_counts)

        while True:
            try:
                message_batch = client.messages.batches.retrieve(
                    message_batch_id=message_batch.id,
                )
            except Exception as e:  # noqa: E722 E841
                logger.warning(f"Error connecting to Anthropic. Retrying in 10s. Exception: {e}")
                pass
            if any(
                [
                    current_request_counts[key] != dict(message_batch.request_counts)[key]
                    for key in current_request_counts
                ]
            ):
                current_request_counts = dict(message_batch.request_counts)
                error_sum = sum([current_request_counts[key] for key in current_request_counts if "succeeded" != key])
                logger.info(
                    f"Succeeded Requests Progress: {current_request_counts['succeeded']}/{len(queries)}. Errors: {error_sum}"
                )
            if message_batch.processing_status == "ended":
                break
            time.sleep(10)

        results = []
        repeat_indices = []

        while True:
            try:
                raw_results = client.messages.batches.results(
                    message_batch_id=message_batch.id,
                )
                break
            except Exception as e:
                logger.error(f"Error connecting to batch Anthropic. Retrying in 10 seconds. Exception: {e}")
                time.sleep(10)

        for i, raw_result in enumerate(raw_results):
            request_logger.log_response(ts=ts, batch_idx=i, response=raw_result.model_dump())
            if raw_result.result.type == "succeeded":
                new_messages = self._get_messages_from_anthropic_content(raw_result.result.message.content)
                conversation = [m.copy() for m in queries[i]] + new_messages
                input_tokens = raw_result.result.message.usage.input_tokens
                output_tokens = raw_result.result.message.usage.output_tokens
                results.append(self.InternalRequestResult(conversation, input_tokens, output_tokens, n_retries=retry_idx))
            else:
                results.append(None)
                repeat_indices.append(i)
                if raw_result.result.type == "errored":
                    logger.error(raw_result.result.error)

        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_results = self._anthropic_batch_processing(repeat_queries, retry_idx + 1)
            for i, result in zip(repeat_indices, repeat_results):
                results[i] = result

        return results

    """
        Case 3: Standard API
    """

    def _run_query_with_retry(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API with retries on failure.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        retry_idx = 0
        total_retries = 0
        start_time = time.time()
        while retry_idx < self.max_retries:
            if self.terminated:
                return None
            try:
                result = self._run_query(idx, query, ignore_tool_calls=ignore_tool_calls)
                result.n_retries += total_retries
                result.time = time.time() - start_time
                time.sleep(self.sleep_after_request)
                return result
            except Exception as e:
                if "Max inner retries reached." in str(e):
                    total_retries += self.max_retries_inner
                elif "rate limit" not in str(e).lower() and "429" not in str(e):
                    total_retries += 1
                logger.error(f"Error in outer retries. Exception: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(self.sleep_on_error)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    retry_idx += 1
                if "violating our usage policy" in str(e).lower():
                    print("Stopping - prompt repeatedly violated usage policy -- ", query)
                    if retry_idx > 3:
                        break
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max outer retries reached.")
        else:
            return None

    def _run_query(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if self.api == "google":
            assert "bcn" in self.model
            # TODO make this code nice; for now quick patch:
            # the only model that does not go to openai compatibility api is bcn
            return self._google_query_with_internal_tools(idx, query)
        if self.api == "openai":
            return self._openai_query_with_tools(idx, query, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "together":
            return self._openai_query_with_tools(idx, query, is_together=True, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "anthropic":
            return self._anthropic_query(idx, query)
        elif self.api == "openrouter":
            return self._openai_query_with_tools(idx, query)

    def _anthropic_query(self, idx, query):
        """Queries the Anthropic API.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.

        Returns:
            InternalRequestResult or None
        """
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        system_message = anthropic.NOT_GIVEN
        if query[0]["role"] == "system":
            system_message = query[0]["content"]
            query = query[1:]
        raw_result = client.messages.create(
            model=self.model, messages=self._drop_cot(query), system=system_message, **self.kwargs
        )

        new_messages = self._get_messages_from_anthropic_content(raw_result.content)
        conversation = [m.copy() for m in query] + new_messages
        input_tokens = raw_result.usage.input_tokens
        output_tokens = raw_result.usage.output_tokens
        return self.InternalRequestResult(conversation, input_tokens, output_tokens)

    def _openai_query_with_tools(self, idx, query, is_together=False, ignore_tool_calls=False):
        """Queries the OpenAI API with tools.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            is_together (bool, optional): Whether to use the Together API. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if is_together:
            client = Together()
        else:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)

        if self.use_openai_responses_api:
            return self._openai_query_responses_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)
        else:
            return self._openai_query_chat_completions_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)

    def _openai_query_responses_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API with the responses API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """

        # Set up tools
        response_tools = []
        for tool_desc in self.tool_descriptions:
            if tool_desc["type"] != "function":
                response_tools.append(tool_desc)
            else:
                response_tools.append({"type": "function", **tool_desc["function"]})
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        elif len(response_tools) == 1 and response_tools[0]["type"] == "code_interpreter":
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        total_retries = 0

        for _ in range(total_max_tool_calls + 1):
            # Inner retry to get a response
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    payload = {
                        "model": self.model,
                        "tools": response_tools,
                        "input": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "timeout": self.timeout,
                        "background": self.background,
                        **self.kwargs,
                    }
                    if self.background:
                        payload["background"] = self.background
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    request_logger.log_request(ts=ts, batch_idx=idx, request=payload, **info)
                    response = client.responses.create(**payload)
                    if self.background:
                        while response.status in {"queued", "in_progress"}:
                            time.sleep(15)
                            response = client.responses.retrieve(response.id)
                    request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump())
                except Exception as e:
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    request_logger.log_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    time.sleep(20)
                    logger.error(f"Got OpenAI error in responses api inner. Exception: {e}")
                    continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            # Update state: token counts and conversation (potentially execute tool calls)
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            # Capture reasoning tokens if available (OpenAI reasoning models)
            if hasattr(response.usage, 'output_tokens_details') and response.usage.output_tokens_details is not None:
                if hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens += response.usage.output_tokens_details.reasoning_tokens or 0

            was_tool_call_executed = False
            for out in response.output:
                if out.type == "message":
                    all_messages = ""  # need all together + ID because API crashes otherwise
                    for c in out.content:
                        if c.type == "output_text":
                            all_messages += f"{c.text}\n\n"
                    conversation.append({"role": "assistant", "content": all_messages, "id": out.id})
                elif out.type == "code_interpreter_call":
                    conversation.append(
                        {
                            "type": "code_interpreter_call",
                            "code": out.code,
                            "id": out.id,
                            "container_id": out.container_id,
                        }
                    )
                elif out.type == "function_call":
                    function_name = out.name
                    arguments = json.loads(out.arguments)
                    tool_func = self.tool_functions[function_name]
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        output = f"Error: Tool call after exceeding max # of tool calls ({max_tool_calls[tool_key]})."
                    else:
                        try:
                            output = tool_func(**arguments)  # EXECUTE
                        except Exception as e:
                            output = f"Error executing tool {function_name}. Exception: {e}"
                    if not isinstance(output, str):
                        additional_cost = output[1]
                        input_tokens += additional_cost["input_tokens"]
                        output_tokens += additional_cost["output_tokens"]
                        output = output[0]
                    conversation.append(
                        {
                            "type": "function_call",
                            "call_id": out.call_id,
                            "arguments": out.arguments,
                            "name": out.name,
                        }
                    )
                    was_tool_call_executed = True
                    nb_executed_tool_calls[tool_key] += 1
                    nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                    detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                    if self.include_max_tool_calls:
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                    else:
                        info = ""
                    conversation.append(
                        {"type": "function_call_output", "call_id": out.call_id, "output": output + info}
                    )
                elif out.type == "reasoning":
                    """
                    API change - each code_interpreter_call now comes with this block beforehand and if you don't
                    keep it it crashes. So we keep it intact (so it does not get stripped in _drop_cot
                    and clean_conversation later will do what the code below used to do).
                    Still need to convert to dict.
                    """
                    reasoning_block = {
                        "id": out.id,
                        "summary": [{"text": b.text, "type": b.type} for b in out.summary],
                        "type": out.type,
                        "content": out.content,
                        "encrypted_content": out.encrypted_content,
                        # "status": out.status,
                    }
                    # NOTE: you get status from OpenAI but if you send it back to them they crash out
                    conversation.append(reasoning_block)
                    # summary = ""
                    # for thought in out.summary:
                    #    if thought.text is not None:
                    #        summary += "<thought>" + "\n" + thought.text + "\n" + "</thought>\n"
                    # conversation.append({"role": "assistant", "type": "cot", "content": summary, "id": out.id})
                else:
                    raise ValueError(f"Unknown output type {out.type}")

            # If nothing was run this was the last iteration, stop
            if not was_tool_call_executed or self.terminated:
                break

        if len(conversation) == len(messages):
            conversation.append({"role": "assistant", "content": ""})
        return self.InternalRequestResult(conversation, input_tokens, output_tokens, n_retries=total_retries, reasoning_tokens=reasoning_tokens)

    def _openai_query_chat_completions_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API using chat completions API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool): Whether to ignore tool calls.

        Returns:
            InternalRequestResult or None
        """

        # Set up tools
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        total_retries = 0
        max_output_tokens = self.kwargs.get(self.max_tokens_param, None)

        # As long as we just had a tool response do another request
        was_tool_call_executed = True
        while was_tool_call_executed and not self.terminated:
            was_tool_call_executed = False
            # Inner retry to get a response
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    kwargs = self.kwargs.copy()
                    kwargs[self.max_tokens_param] = max_output_tokens
                    payload = {
                        "model": self.model,
                        "messages": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "tools": self.tool_descriptions if len(self.tool_descriptions) > 0 else None,
                        "timeout": self.timeout,
                        **kwargs,
                    }
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    request_logger.log_request(ts=ts, batch_idx=idx, request=payload, **info)
                    response = client.chat.completions.create(**payload)
                    request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump())
                except Exception as e:
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    request_logger.log_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    if isinstance(e, RateLimitError):
                        logger.info(f"Got OpenAI CC rate limit error. Sleeping for 60 seconds. Exception: {e}")
                        time.sleep(60)
                        continue
                    else:
                        if "maximum context length" in str(e).lower() or "input token count" in str(e).lower():
                            max_output_tokens = max_output_tokens // 2
                            logger.info(
                                f"Got OpenAI CC max context length error. Reducing max output tokens to {max_output_tokens} and retrying. Exception: {e}"
                            )
                        logger.info(f"Got OpenAI CC non ratelimit error. Sleeping for 20 seconds: {e}")
                        time.sleep(20)
                        continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            # Update state: token counts and conversation (potentially execute tool calls)
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.total_tokens - response.usage.prompt_tokens
            # Capture reasoning tokens if available (OpenAI reasoning models via chat completions)
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details is not None:
                if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens += response.usage.completion_tokens_details.reasoning_tokens or 0
            message = response.choices[0].message
            if self.context_limit is not None:
                max_output_tokens = self.context_limit
                if max_output_tokens is not None:
                    max_output_tokens -= response.usage.total_tokens
                    max_output_tokens = min(max_output_tokens, 
                                            self.kwargs.get(self.max_tokens_param, float("inf")))
            else:
                max_output_tokens = self.kwargs.get(self.max_tokens_param, None)
            # Add CoT and rest of message separately
            # TODO: if we notice new ways to return CoT they should be mangled here. Likely missing something.
            if hasattr(message, "reasoning_details") and message.reasoning_details:
                for detail in message.reasoning_details:
                    if isinstance(detail, dict):
                        type_ = detail.get("type", "")
                        text_ = detail.get("text", "")
                        sumary_ = detail.get("sumary", "")
                    else:
                        type_ = detail.type
                        text_ = detail.text
                        sumary_ = detail.sumary
                    if type_ == "reasoning.text":
                        conversation.append({"role": "assistant", "type": "cot", "content": text_})
                    elif type_ == "reasoning.summary":
                        conversation.append({"role": "assistant", "type": "cot", "content": sumary_})
                    else:
                        pass  # encrypted
            else:
                # Reasoning details trumps others as there is repetition
                if hasattr(message, "reasoning") and message.reasoning:
                    conversation.append({"role": "assistant", "type": "cot", "content": message.reasoning})
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    conversation.append({"role": "assistant", "type": "cot", "content": message.reasoning_content})

            message_dict = message.model_dump()
            message_dict = {k: v for k, v in message_dict.items() if v is not None}  # Drop nulls
            if "reasoning" in message_dict:
                del message_dict["reasoning"]
            if "reasoning_content" in message_dict and "deepseek" not in self.model.lower():
                del message_dict["reasoning_content"]
            if "reasoning_details" in message_dict:
                pass  # Models that have this like to also have it back so it's kept, as "type":"cot" blocks get filtered out before the API. Normalize_conversation does the opposite: ignores this but keeps clean "cot" blocks.
            conversation.append(message_dict)  # Should have tool calls inside too; and reasoning_details

            # Try to execute all tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        function_name = tool_call.get("function", dict()).get("name", "")
                    else:
                        function_name = tool_call.function.name
                    if function_name not in self.tool_functions:
                        logger.warning(f"Tool {function_name} not found, skipping.")
                        continue
                    tool_func = self.tool_functions[function_name]

                    # If no budget return error
                    # NOTE: just erroring out here might stop the request loop but the model will be given last chance.
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        error = f"Error: Exceeded maximum number of tool calls ({max_tool_calls[tool_key]})."
                        conversation.append({"role": "tool", "tool_call_id": tool_call.id, "content": error})
                    else:
                        # Execute tool
                        arguments = json.loads(tool_call.function.arguments)
                        try:
                            output = tool_func(**arguments)
                        except Exception as e:
                            output = f"Error executing tool {function_name}. Exception: {e}"
                        # Tools can return additional cost
                        if isinstance(output, tuple):
                            output, extra_cost = output
                            input_tokens += extra_cost["input_tokens"]
                            output_tokens += extra_cost["output_tokens"]

                        nb_executed_tool_calls[tool_key] += 1

                        nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                        detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                        conversation.append(
                            {
                                "role": "tool",
                                "tool_name": function_name,
                                "tool_call_id": tool_call.id,
                                "content": output + info,
                            }
                        )

                    was_tool_call_executed = True

        if total_max_tool_calls > 0:
            logger.info(f"Finished on a loop without tool calls, after executing {nb_executed_tool_calls} calls total.")

        return self.InternalRequestResult(conversation, input_tokens, output_tokens, n_retries=total_retries, reasoning_tokens=reasoning_tokens)

    def _google_query_with_internal_tools(self, idx, messages):
        """Queries Google for BCN.
        InternalRequestResult or None
        """
        # NOTE: expect single turn (since internal tool calls) and don't reprompt
        # TODO implement properly or find a way to use openai compat
        assert len(messages) == 1 and messages[0]["role"] == "user"

        conversation = [m.copy() for m in messages]

        headers = {
            "x-goog-api-key": f"{self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [{"role": "user", "parts": {"text": messages[0]["content"]}}],
            "tools": self.tool_descriptions,
        }

        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{datetime.now().microsecond:06d}"
        request_logger.log_request(ts=ts, batch_idx=idx, request=payload)

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
        )
        request_logger.log_response(ts=ts, batch_idx=idx, response=response.json())

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        json_response = response.json()

        if "candidates" not in json_response:
            raise Exception(f"Error: {json_response}")

        message = json_response["candidates"][0]["content"]
        parts = message["parts"]
        role = message["role"]
        assert role == "model"

        input_tokens = json_response["usageMetadata"]["promptTokenCount"]
        output_tokens = json_response["usageMetadata"]["candidatesTokenCount"]

        def _clear_buffer(conversation, buffer, mode):
            if buffer is None or len(buffer) == 0:
                return
            if mode == "cot":
                conversation.append({"role": "assistant", "type": "cot", "content": buffer})
            elif mode == "response":
                conversation.append({"role": "assistant", "type": "response", "content": buffer})
            else:
                return

        mode = None
        buffer = ""
        # TODO: We will log these as external tool call now for speed, transpile later
        for part in parts:
            if "thought" in part and part["thought"]:
                if mode != "cot":
                    _clear_buffer(conversation, buffer, mode)
                    buffer = ""
                    mode = "cot"
                buffer += f"{part['text']}"
            elif "functionCall" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                conversation.append(
                    {
                        "type": "tool_call",
                        "role": "assistant",
                        "call_id": "1",
                        "arguments": part["functionCall"]["args"],
                        "name": part["functionCall"]["name"],
                    }
                )
            elif "functionResponse" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                name = part["functionResponse"]["name"]
                resp = part["functionResponse"]["response"]
                if "result" in resp:
                    result = resp["result"]
                elif "content" in resp:
                    result = resp["content"]
                else:
                    result = ""
                conversation.append({"role": "tool", "tool_name": f"{name}", "tool_call_id": "1", "content": result})
            elif "text" in part:
                if mode != "response":
                    _clear_buffer(conversation, buffer, mode)
                    buffer = ""
                    mode = "response"
                buffer += f"{part['text']}"
        _clear_buffer(conversation, buffer, mode)

        return self.InternalRequestResult(
            conversation=conversation, input_tokens=input_tokens, output_tokens=output_tokens
        )
