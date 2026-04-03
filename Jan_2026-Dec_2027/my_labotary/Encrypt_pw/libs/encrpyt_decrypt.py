import os
import cryptography
import streamlit as st

# ===============================================================
# Level 1 . 
#       Only focus on one-one mapping numeric to symbolic
# ===============================================================
map_dict = {
    '1': '!',
    '2': '@',
    '3': "#",
    '4': "$",
    '5': "%",
    '6': "^",
    '7': "&",
    '8': "*",
    '9': "(",
    '0': ")"
}

def encrypt_numeric_mapping(text):
    result = ''.join(map_dict.get(ch, ch) for ch in text)
    st.write("### Output")
    st.success(result)

def decrypt_symbolic_mapping(text):
    # reverse mapping
    rev_map = {v: k for k, v in map_dict.items()}
    result = ''.join(rev_map.get(ch, ch) for ch in text)
    st.write("### Output")
    st.success(result)

# ===============================================================
# Level 2 . 
#       One-one mapping by permutation the secret-indexes
# ===============================================================
import random

def encrypt_shuffle(s, key):
    idx = list(range(len(s)))
    random.seed(key)
    random.shuffle(idx)
    st.write("### Output")

    st.success(''.join(s[i] for i in idx))
    st.success(f"Indexes: {idx}")

def decrypt_shuffle(cipher, idx):
    res = [''] * len(cipher)
    for i, j in enumerate(idx):
        res[j] = cipher[i]
    st.write("### Output")
    st.success(''.join(res))

# ===============================================================
# Level 3. 
#       Give description here
# ===============================================================
import string

def encrypt_substitution(s, key):
    random.seed(key)
    alphabet = string.printable
    shuffled = list(alphabet)
    random.shuffle(shuffled)
    table = dict(zip(alphabet, shuffled))
    return ''.join(table[c] for c in s)


def decrypt_substitution(cipher, key):
    random.seed(key)
    alphabet = string.printable
    shuffled = list(alphabet)
    random.shuffle(shuffled)
    table = dict(zip(shuffled, alphabet))
    return ''.join(table[c] for c in cipher)

# ===============================================================
# Level 4. 
#       Give description here
# ===============================================================
def _prepare_vigenere_key(key) -> str:
    """
    Prepares the Vigenere key by ensuring it's a string.
    Handles int, float, str, and bytes types.
    """
    if isinstance(key, (int, float)):
        key_str = str(key)
    elif isinstance(key, str):
        key_str = key
    elif isinstance(key, bytes):
        # Assuming bytes keys are intended to be decoded text
        key_str = key.decode('utf-8')
    else:
        raise TypeError(f"Unsupported key type: {type(key)}. Key must be str, int, float, or bytes.")

    if not key_str:
        raise ValueError("Vigenere key cannot be an empty string after conversion.")
    return key_str

def encrypt_vigenere(s: str, key) -> str:
    key_str = _prepare_vigenere_key(key)
    return ''.join(
        chr((ord(c) + ord(key_str[i % len(key_str)])) % 256)
        for i, c in enumerate(s)
    )

def decrypt_vigenere(c: str, key) -> str:
    key_str = _prepare_vigenere_key(key)
    return ''.join(
        chr((ord(ch) - ord(key_str[i % len(key_str)])) % 256)
        for i, ch in enumerate(c)
    )

# ===============================================================
# Level 5. 
#       Give description here
# ===============================================================
def encrypt_stream_weak(data: bytes, seed: int) -> bytes:
    rnd = random.Random(seed)
    text_bytes = data.encode('utf-8')
    res = bytes(b ^ rnd.getrandbits(8) for b in data)
    st.write(res)


def decrypt_stream_weak(cipher: bytes, seed: int) -> bytes:
    res = encrypt_stream_weak(cipher, seed)
    st.write(res.decode('utf-8'))

# ===============================================================
# Level 6. 
#       Give description here
# ===============================================================
def otp_encrypt(data: bytes, key: bytes) -> bytes:
    data = data.encode('utf-8')
    assert len(data) == len(key), f"Data length ({len(data)}) must match key length ({len(key)})"
    res = bytes(d ^ k for d, k in zip(data, key))
    st.write(res)

def otp_decrypt(cipher: bytes, key: bytes) -> bytes:
    res = otp_encrypt(cipher, key)
    st.write(res.decode('utf-8'))

# ===============================================================
# Level 7. 
#       Give description here
# ===============================================================
from hashlib import sha256

def xor_cipher(data: bytes, key: bytes) -> bytes:
    return bytes(d ^ key[i % len(key)] for i, d in enumerate(data))

def derive_key_from_password(password: str, length: int) -> bytes:
    # password argument expects a string, then it encodes it to bytes
    digest = sha256(password.encode()).digest()
    return (digest * (length // len(digest) + 1))[:length]

def encrypt_password_based(data: bytes, password: str) -> bytes:
    key = derive_key_from_password(password, len(data))
    res = xor_cipher(data, key)
    st.write(res)

def decrypt_password_based(cipher: bytes, password: str) -> bytes:
    # This function expects 'password' to be a string
    res = encrypt_password_based(cipher, password)
    st.write(res.decode('utf-8'))

# ===============================================================
# Level 8. 
#       Give description here
# ===============================================================
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Generate a new, appropriately sized key for AESGCM (e.g., 16 bytes for 128-bit AES)
aes_gcm_key = os.urandom(16)

def encrypt_aes_gcm(data: bytes, key: bytes) -> bytes:
    aes = AESGCM(key)
    nonce = os.urandom(12)
    res = nonce + aes.encrypt(nonce, data, None)
    st.write(res)

def decrypt_aes_gcm(cipher: bytes, key: bytes) -> bytes:
    nonce, ct = cipher[:12], cipher[12:]
    aes = AESGCM(key)
    res = aes.decrypt(nonce, ct, None)
    st.write(res.decode('utf-8'))