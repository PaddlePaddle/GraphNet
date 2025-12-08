import hashlib


def get_sha256_hash(content):
    m = hashlib.sha256()
    m.update(content.encode())
    return m.hexdigest()
