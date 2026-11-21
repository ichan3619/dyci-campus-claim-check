"""
Saved passage vectors (scale plan S1, docs/plans/scale_plan.md; approved 2026-10-25).

CAMPUS_VERIFY_VECTOR_STORE=<folder> (default unset = off): src/api/main.py then builds a PersistentE5Encoder instead of
the plain E5Encoder (src/retrieval/e5_retriever.py, frozen, not edited). A restart reads the vectors instead of
embedding every passage again (minutes at thousands of posts).

WHY THE KEY IS THE WHOLE BATCH, NOT ONE PASSAGE
    E5 vectors depend on which texts were embedded together: padding changes the last bits (measured 2026-10-25: up to
    6.7e-8 between a passage embedded alone and in a batch of 40). Scores are returned at full precision, so a store
    keyed by passage would change answers. E5Encoder calls _encode() with exactly the batch of passages it is missing;
    this class stores that batch's matrix under sha256(the batch, in order) and returns it unchanged on a later
    identical call. A restart therefore reproduces a fresh build bit for bit.

WHAT IS STORED, AND WHEN
    Passage batches only ("passage: " prefix; queries are cheap and never stored), and only while a corpus reload is
    building the index (persisting(), set by src/api/background_reload.py). Pages fetched by /verify/url are never
    written to disk.

FOLDER AND FILES
    <root>/<signature hash>/signature.json: everything that changes the vectors (model name, revision or weights hash,
    prefix, normalisation, max_seq_length, modules, precision, device, library versions). A different signature is a
    different folder, so vectors from two models are never mixed.
    seg-<uuid>.npz: one write per reload (keys, starts, counts, vectors), written to a temporary name and renamed
    (atomic on one drive), never modified afterwards. Safe across threads and processes; a startup scan rebuilds the
    index. An unreadable segment is skipped and reported; any store error falls back to computing in memory.
"""
import contextlib
import hashlib
import json
import logging
import os
import threading
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.retrieval.e5_retriever import PASSAGE_PREFIX, E5Encoder

logger = logging.getLogger(__name__)

ENV = "CAMPUS_VERIFY_VECTOR_STORE"
_state = threading.local()


@contextlib.contextmanager
def persisting():
    """Within this block (in this thread only), newly computed passage batches are saved to the store."""
    previous = getattr(_state, "on", False)
    _state.on = True
    try:
        yield
    finally:
        _state.on = previous


def _persisting() -> bool:
    return getattr(_state, "on", False)



def batch_key(texts: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_revision(model_name: str) -> Optional[str]:
    """The Hugging Face cache revision for a hub id, or the weights' sha256 for a local folder; None when unknown."""
    if os.path.isdir(model_name):
        for weights in ("model.safetensors", "pytorch_model.bin"):
            path = os.path.join(model_name, weights)
            if os.path.isfile(path):
                return "sha256:" + _file_sha256(path)
        return None
    try:
        from huggingface_hub import constants
        ref = os.path.join(constants.HF_HUB_CACHE, "models--" + model_name.replace("/", "--"), "refs", "main")
        with open(ref, encoding="utf-8") as f:
            return f.read().strip() or None
    except Exception:
        return None


def model_signature(encoder: E5Encoder, revision: Optional[str] = None) -> Optional[dict]:
    """Everything that changes the vectors, or None when the model revision cannot be established."""
    revision = revision or model_revision(encoder.model_name)
    if not revision:
        return None
    model = encoder._model
    try:
        modules = [type(m).__name__ for m in model]
    except TypeError:
        modules = [type(model).__name__]
    versions = {}
    for lib in ("torch", "transformers", "sentence_transformers", "numpy"):
        try:
            versions[lib] = __import__(lib).__version__
        except Exception:
            versions[lib] = None
    return {"model": encoder.model_name, "revision": revision, "prefix": PASSAGE_PREFIX, "normalize_embeddings": True,
            "max_seq_length": getattr(model, "max_seq_length", None), "modules": modules, "precision": "float32",
            "truncate_dim": None, "device": str(getattr(model, "device", "cpu")), "versions": versions}


class SegmentStore:
    """Batch key -> matrix, read from and appended to segment files in one signature folder."""

    def __init__(self, folder: str, signature: dict):
        self.folder, self.errors = folder, []
        self._lock = threading.Lock()
        self._vectors: Dict[str, np.ndarray] = {}
        self._pending: List[tuple] = []
        os.makedirs(folder, exist_ok=True)
        sig_path = os.path.join(folder, "signature.json")
        if not os.path.exists(sig_path):
            self._atomic_write(sig_path, json.dumps(signature, indent=1, sort_keys=True).encode("utf-8"))
        for name in sorted(os.listdir(folder)):
            if name.startswith("seg-") and name.endswith(".npz"):
                self._load(os.path.join(folder, name))

    def _load(self, path: str) -> None:
        try:
            with np.load(path, allow_pickle=False) as z:
                keys, starts, counts, vectors = z["keys"], z["starts"], z["counts"], z["vectors"]
            for k, s, c in zip(keys, starts, counts):
                self._vectors[str(k)] = vectors[int(s):int(s) + int(c)]
        except Exception as e:                           # skip it: those batches are simply computed again
            self.errors.append(f"unreadable segment {os.path.basename(path)}: {e}")
            logger.warning("Vector store: skipping %s: %s", path, e)

    def __len__(self) -> int:
        return len(self._vectors)

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._vectors.get(key)

    def put(self, key: str, matrix: np.ndarray) -> None:
        with self._lock:
            if key not in self._vectors:
                self._vectors[key] = matrix
                self._pending.append((key, matrix))

    def flush(self) -> Optional[str]:
        """Write the batches added since the last flush to one new segment. Returns an error message, or None."""
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return None
        counts = [m.shape[0] for _, m in pending]
        starts = np.cumsum([0] + counts[:-1])
        data = {"keys": np.array([k for k, _ in pending], dtype="U64"), "starts": np.array(starts, dtype=np.int64),
                "counts": np.array(counts, dtype=np.int64), "vectors": np.concatenate([m for _, m in pending])}
        path = os.path.join(self.folder, f"seg-{uuid.uuid4().hex}.npz")
        try:
            import io
            buf = io.BytesIO()
            np.savez(buf, **data)
            self._atomic_write(path, buf.getvalue())
            return None
        except Exception as e:                           # the vectors stay in memory for this process
            message = f"vector store write failed: {e}"
            self.errors.append(message)
            logger.warning("Vector store: %s", message)
            return message

    @staticmethod
    def _atomic_write(path: str, data: bytes) -> None:
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in (1, 2):                           # Windows: antivirus can hold a new file for a moment
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 2:
                    with contextlib.suppress(OSError):
                        os.remove(tmp)
                    raise
                import time
                time.sleep(0.2)


def store_folder(root: str, signature: dict) -> str:
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return os.path.join(root, f"{signature['model'].replace('/', '--').replace(os.sep, '--')[-60:]}-{digest}")


class LockedE5Encoder(E5Encoder):
    """E5Encoder whose caches are used under one lock. E5Encoder's OrderedDict caches are not thread-safe, and with
    background reloads a rebuild thread and request threads embed at the same time. The lock is held for one call
    (one document's passages, or one query), never for a whole rebuild."""

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu", model=None):
        super().__init__(model_name, device, model)
        self._lock = threading.RLock()

    def encode_passages(self, texts: Sequence[str]):
        with self._lock:
            return super().encode_passages(texts)

    def encode_queries(self, texts: Sequence[str]):
        with self._lock:
            return super().encode_queries(texts)


class PersistentE5Encoder(LockedE5Encoder):
    """LockedE5Encoder with saved passage batches."""

    def __init__(self, store_root: str, model_name: Optional[str] = None, device: str = "cpu", model=None,
                 revision: Optional[str] = None):
        super().__init__(model_name, device, model)
        self.store: Optional[SegmentStore] = None
        self.store_error: Optional[str] = None
        try:
            signature = model_signature(self, revision)
            if signature is None:
                raise RuntimeError(f"model revision for {self.model_name!r} unknown; vectors are not saved")
            self.store = SegmentStore(store_folder(store_root, signature), signature)
            if self.store.errors:
                self.store_error = "; ".join(self.store.errors)
        except Exception as e:                           # fall back to the plain in-memory encoder
            self.store_error = f"vector store off: {e}"
            logger.warning("Vector store: %s", self.store_error)

    def _encode(self, texts: Sequence[str]):
        texts = list(texts)
        if self.store is None or not texts or not all(t.startswith(PASSAGE_PREFIX) for t in texts):
            return super()._encode(texts)
        key = batch_key(texts)
        hit = self.store.get(key)
        if hit is not None and hit.shape[0] == len(texts):
            return hit.copy()
        vectors = super()._encode(texts)
        if _persisting():
            self.store.put(key, np.array(vectors, copy=True))
        return vectors

    def flush(self) -> Optional[str]:
        if self.store is None:
            return None
        error = self.store.flush()
        if error:
            self.store_error = error
        return error
