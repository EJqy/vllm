"""vLLM general plugin: grammar-aware speculative drafting for structured
outputs. Enabled with GRAMMAR_DRAFT=1; see `config.py` for the switches."""

import importlib.abc
import importlib.util
import sys
from collections.abc import Callable

from .config import GrammarDraftConfig

_installed = False


def register() -> None:
    """Entry point (`vllm.general_plugins`). Idempotent; a no-op when off."""
    global _installed
    cfg = GrammarDraftConfig.from_env()
    if _installed or not cfg.enabled:
        return
    _installed = True

    def scheduler(_module):
        from . import scheduler_hook

        scheduler_hook.install(cfg)

    def worker(_module):
        from . import worker_hook

        worker_hook.install()

    # Patch lazily so a process only imports what it already uses: the API
    # server imports neither module, the engine core imports both.
    _after_import("vllm.v1.structured_output", scheduler)
    _after_import("vllm.v1.worker.gpu.model_runner", worker)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs a callback right after the named module finishes executing."""

    def __init__(self, name: str, callback: Callable):
        self.name, self.callback = name, callback

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            self.callback(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def _after_import(name: str, callback: Callable) -> None:
    if name in sys.modules:
        callback(sys.modules[name])
    else:
        sys.meta_path.insert(0, _PostImportFinder(name, callback))
