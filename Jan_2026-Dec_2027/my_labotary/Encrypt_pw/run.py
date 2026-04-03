import streamlit as st
from libs.encrpyt_decrypt import *

st.set_page_config(layout="wide")
st.title("PW Encrypted / Decrypted")
st.write("--------------------")

level_sel = st.selectbox("Select level", [f"Level {idx}" for idx in range(1, 9)])

# ================= DESCRIPTION =================
descriptions = {
    "Level 1": "Simple numeric → symbol mapping.",
    "Level 2": "Shuffle string using seeded permutation.",
    "Level 3": "Random substitution cipher (based on printable chars).",
    "Level 4": "Vigenère cipher (byte-wise).",
    "Level 5": "Weak stream cipher using random XOR.",
    "Level 6": "One-Time Pad (OTP) encryption.",
    "Level 7": "Password-based XOR cipher using SHA256.",
    "Level 8": "AES-GCM authenticated encryption."
}

c0, _, c1, _, c2 = st.columns([1, 0.1, 1, 0.1, 1])

# ================= DESCRIPTION =================
with c0:
    st.write("#### Description")
    st.info(descriptions[level_sel])

# ================= ENCRYPT =================
with c1:
    st.write("#### Encrypt")
    inp = st.text_input("Input text", key="enc_text")

    if level_sel == "Level 1":
        if st.button("Encrypt L1"):
            encrypt_numeric_mapping(inp)

    elif level_sel == "Level 2":
        key = st.number_input("Seed", value=53)
        if st.button("Encrypt L2"):
            encrypt_shuffle(inp, key)

    elif level_sel == "Level 3":
        key = st.text_input("Key")
        if st.button("Encrypt L3"):
            res = encrypt_substitution(inp, key)
            st.success(res)

    elif level_sel == "Level 4":
        key = st.text_input("Key")
        if st.button("Encrypt L4"):
            res = encrypt_vigenere(inp, key)
            st.success(res)

    elif level_sel == "Level 5":
        seed = st.number_input("Seed", value=123)
        if st.button("Encrypt L5"):
            res = encrypt_stream_weak(inp.encode(), seed)
            st.write(res)

    elif level_sel == "Level 6":
        key = st.text_input("Key (same length)")
        if st.button("Encrypt L6"):
            res = otp_encrypt(inp, key.encode())
            st.write(res)

    elif level_sel == "Level 7":
        password = st.text_input("Password")
        if st.button("Encrypt L7"):
            res = encrypt_password_based(inp.encode(), password)
            st.write(res)

    elif level_sel == "Level 8":
        key = st.text_input("Key (16 bytes)")
        if st.button("Encrypt L8"):
            res = encrypt_aes_gcm(inp.encode(), key.encode())
            st.write(res)

# ================= DECRYPT =================
with c2:
    st.write("#### Decrypt")
    inp = st.text_input("Input cipher", key="dec_text")

    if level_sel == "Level 1":
        if st.button("Decrypt L1"):
            decrypt_symbolic_mapping(inp)

    elif level_sel == "Level 2":
        idx = st.text_area("Indexes (comma separated)")
        if st.button("Decrypt L2"):
            idx_list = list(map(int, idx.split(",")))
            decrypt_shuffle(inp, idx_list)

    elif level_sel == "Level 3":
        key = st.text_input("Key", key="k3d")
        if st.button("Decrypt L3"):
            res = decrypt_substitution(inp, key)
            st.success(res)

    elif level_sel == "Level 4":
        key = st.text_input("Key", key="k4d")
        if st.button("Decrypt L4"):
            res = decrypt_vigenere(inp, key)
            st.success(res)

    elif level_sel == "Level 5":
        seed = st.number_input("Seed", value=123, key="s5d")
        if st.button("Decrypt L5"):
            res = decrypt_stream_weak(inp.encode(), seed)

    elif level_sel == "Level 6":
        key = st.text_input("Key", key="k6d")
        if st.button("Decrypt L6"):
            res = otp_decrypt(inp.encode(), key.encode())

    elif level_sel == "Level 7":
        password = st.text_input("Password", key="p7d")
        if st.button("Decrypt L7"):
            res = decrypt_password_based(inp.encode(), password)

    elif level_sel == "Level 8":
        key = st.text_input("Key", key="k8d")
        if st.button("Decrypt L8"):
            res = decrypt_aes_gcm(inp.encode(), key.encode())