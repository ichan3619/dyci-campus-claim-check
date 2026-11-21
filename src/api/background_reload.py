"""
Corpus reloads for large collections (scale plan S1, docs/plans/scale_plan.md; approved 2026-10-25).

ScalablePipelineManager extends PipelineManager (src/api/pipeline.py, frozen, not edited). It reuses the parent's
reload, build and swap unchanged and adds two things:
1. Every index build runs inside vector_store.persisting(), and the new passage batches are flushed to the store
   afterwards (CAMPUS_VERIFY_VECTOR_STORE). That applies to every build path: startup, reload(), the in-request reload
   in current(), and the Facebook refresh.
2. With background=True (CAMPUS_VERIFY_BACKGROUND_RELOAD=1), current() never builds. When a corpus file has changed,
   it starts one background thread running the parent's reload() and returns the current pipeline at once. The
   parent swaps the pipeline in a single assignment at the end of a successful build, and requests never wait on its
   lock (they only try it without blocking), so no request waits for a rebuild. A failed rebuild keeps the previous
   pipeline and is reported in reload_error, as before.
3. While a background rebuild runs, Python switches threads every REBUILD_SWITCH_INTERVAL seconds instead of its
   default 5 ms, and the previous value is restored afterwards. Reason (S1 gate 3 diagnosis, 2026-10-25): the rebuild
   is mostly pure-Python work, and at the default interval a request thread that keeps releasing the interpreter lock
   (around every model call) waits up to 5 ms each time to get it back. Measured on 3,000 synthetic posts: one
   request took the whole rebuild (1.96 s) at the default, and 23 requests ran at most 0.467 s at 0.5 ms; the rebuild
   itself took 5.0 s instead of 1.9 s. A pure-Python busy thread showed the same effect, so it is not this module's
   locks.
"""
import logging
import sys
import threading
from typing import Optional

from src.api.pipeline import Pipeline, PipelineManager, _now
from src.retrieval import vector_store

logger = logging.getLogger(__name__)

BACKGROUND_ENV = "CAMPUS_VERIFY_BACKGROUND_RELOAD"
REBUILD_SWITCH_INTERVAL = 0.0005


class ScalablePipelineManager(PipelineManager):
    def __init__(self, *args, background: bool = False, **kwargs):
        self._background = background
        self._rebuild_thread: Optional[threading.Thread] = None
        self._rebuild_guard = threading.Lock()
        super().__init__(*args, **kwargs)                # runs the first build through _reload_locked below

    def _reload_locked(self) -> bool:
        with vector_store.persisting():
            built = super()._reload_locked()
        flush = getattr(self.encoder, "flush", None)
        if flush is not None:
            error = flush()
            if error and built:                          # keep serving; say why the vectors were not saved
                self.reload_error = f"{_now()}: {error}"
        return built

    def current(self) -> Pipeline:
        if not self._background:
            return super().current()
        if self.settings.reload_seconds >= 0 and self._clock() - self._last_check >= self.settings.reload_seconds:
            self._last_check = self._clock()
            if self._fingerprint_now() != self._fingerprint:
                self._start_rebuild()
        return self._pipeline

    def _start_rebuild(self) -> None:
        with self._rebuild_guard:
            if self._rebuild_thread is not None and self._rebuild_thread.is_alive():
                return                                   # one rebuild at a time; the next check catches later changes
            self._rebuild_thread = threading.Thread(target=self._rebuild, name="corpus-rebuild", daemon=True)
            self._rebuild_thread.start()

    def _rebuild(self) -> None:
        previous = sys.getswitchinterval()
        sys.setswitchinterval(REBUILD_SWITCH_INTERVAL)   # requests keep their turn while this thread builds (see 3.)
        try:
            self.reload()
        except Exception as e:                           # never let the thread die silently
            self.reload_error = f"{_now()}: {e}"
            logger.warning("Background corpus rebuild failed: %s", e)
        finally:
            sys.setswitchinterval(previous)

    def wait_for_rebuild(self, timeout: Optional[float] = None) -> bool:
        """True when no rebuild is running (after waiting up to `timeout`). For tests and tools."""
        thread = self._rebuild_thread
        if thread is not None:
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def stop(self) -> None:
        super().stop()
        self.wait_for_rebuild(timeout=5)
