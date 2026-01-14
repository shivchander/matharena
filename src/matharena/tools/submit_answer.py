"""
Hash matching for answers that need to be kept secret (e.g., Project Euler).
"""

import hashlib


def check_hash_match(answer: str, hash_value: str) -> bool:
    """
    Check if the answer matches the stored hash.

    Args:
        answer: The model's answer as a string
        hash_value: The expected hash value

    Returns:
        True if the answer's hash matches the expected hash
    """
    # Try different hash formats
    answer_clean = answer.strip()

    # Try SHA256
    sha256_hash = hashlib.sha256(answer_clean.encode()).hexdigest()
    if sha256_hash == hash_value or sha256_hash.startswith(hash_value):
        return True

    # Try MD5
    md5_hash = hashlib.md5(answer_clean.encode()).hexdigest()
    if md5_hash == hash_value or md5_hash.startswith(hash_value):
        return True

    return False
