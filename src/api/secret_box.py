"""
Encryption at rest for the Facebook page tokens that administrators enter (src/api/sources.py).

    CAMPUS_VERIFY_SOURCES_KEY set   Fernet (AES-128-CBC + HMAC-SHA256, the `cryptography` package already installed)
                                    with that key. Make one with: python -c "from cryptography.fernet import Fernet;
                                    print(Fernet.generate_key().decode())", and keep it outside the repository.
    unset, on Windows               DPAPI (CryptProtectData), tied to the Windows account running the checker: no key
                                    file exists, and a copy of the database is unreadable on another account or PC.
    unset, elsewhere                refused: tokens are never stored in plain text.

The database never holds a token in plain text, and nothing here logs or returns one.
"""
import ctypes
import os
from typing import Optional

KEY_ENV = "CAMPUS_VERIFY_SOURCES_KEY"
_ENTROPY = b"campus-verify-sources-v1"


class SecretBoxError(RuntimeError):
    pass


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes):
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf


def _dpapi(data: bytes, protect: bool) -> bytes:
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    src, _keep = _blob(data)
    ent, _keep2 = _blob(_ENTROPY)
    out = _Blob()
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(src), None, ctypes.byref(ent), None, None, 0x1, ctypes.byref(out))      # 0x1: never show UI
    if not ok:
        raise SecretBoxError("Windows could not " + ("protect" if protect else "unprotect") + " the token")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


class SecretBox:
    def __init__(self, key: Optional[str] = None):
        key = key if key is not None else (os.environ.get(KEY_ENV, "").strip() or None)
        if key:
            from cryptography.fernet import Fernet
            try:
                self._fernet = Fernet(key.encode("ascii"))
            except (ValueError, UnicodeEncodeError):
                raise SecretBoxError(f"{KEY_ENV} is not a valid Fernet key") from None
            self.method = "key"
        elif os.name == "nt":
            self._fernet = None
            self.method = "windows-dpapi"
        else:
            raise SecretBoxError(f"Set {KEY_ENV} to store page tokens on this system")

    def protect(self, secret: str) -> bytes:
        data = secret.encode("utf-8")
        return b"F" + self._fernet.encrypt(data) if self._fernet else b"D" + _dpapi(data, True)

    def unprotect(self, blob: bytes) -> str:
        kind, body = blob[:1], blob[1:]
        if kind == b"F":
            if not self._fernet:
                raise SecretBoxError(f"this token was stored with {KEY_ENV}; set it again to use the token")
            from cryptography.fernet import InvalidToken
            try:
                return self._fernet.decrypt(body).decode("utf-8")
            except InvalidToken:
                raise SecretBoxError(f"this token cannot be read with the current {KEY_ENV}") from None
        if kind == b"D":
            if os.name != "nt":
                raise SecretBoxError("this token was protected by Windows on another system")
            return _dpapi(body, False).decode("utf-8")
        raise SecretBoxError("unknown token format")
