import threading
from contextlib import contextmanager
from typing import Any, Tuple

import torch
from typing_extensions import Self
from ..._c.py_mount import mount, unmount
from ... import CONFIG


_mounted = False
_execution_depth = 0
_execution_lock = threading.Lock()


def _ensure_mounted():
    """Mount Object.save as the universal `.save` method.

    Lazy one-time setup. Called from .save() / nnsight.save() so we only
    pay the C-level mount cost once, on first use.
    """
    global _mounted
    if CONFIG.APP.PYMOUNT and not _mounted:
        mount(Object.save, "save")
        _mounted = True


@contextmanager
def _mounted_during_execution():
    """Mount ``Object.save`` while a trace is executing and unmount it once none is.

    ``_ensure_mounted`` installs ``save`` on ``object`` itself, so while it's mounted, ``save`` shows up in
    ``dir()`` of every class. Leaving it there after the trace breaks libraries that inspect class
    attributes: e.g. anyio's ``TypedAttributeSet`` raises "Attribute 'save' is missing its type annotation"
    when ``anyio.streams.tls`` is first imported (which the OpenAI client does lazily). Traces can nest (e.g.
    inside a session) or run from several threads, so this is reference counted under a lock; ``save`` is
    re-mounted on the next trace.
    """
    global _execution_depth, _mounted
    with _execution_lock:
        _execution_depth += 1
        _ensure_mounted()
    try:
        yield
    finally:
        with _execution_lock:
            _execution_depth -= 1
            if _execution_depth == 0 and _mounted:
                unmount("save")
                _mounted = False


def save(object: Any):

    Globals.saves.add(id(object))

    return object


class Object(torch.Tensor):

    def save(self, _=0):
        """
        Save an object to be accessable after the trace context is exited.

        Examples:

        >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
        >>> with model.trace("Hello World"):
        ...     attn_0 = model.transformer.h[0].attn.output.save()
        >>> print(attn_0)
        """

        save(self)

        return self

    def __getattr__(self, name: str) -> Self:

        return super().__getattr__(name)

    def __getitem__(self, key: Any) -> Self:

        return super().__getitem__(key)

    def __call__(self, *args: Any, **kwargs: Any) -> Self:

        return super().__call__(*args, **kwargs)


class TracingCache:

    def __init__(self):
        self.cache = {}
        self.code_cache = {}

    def get(self, cache_key: Tuple):
        """
        Check if the given filename and lineno is in the cache.
        """
        return self.cache.get(cache_key, None)

    def add(self, cache_key: Tuple, value: Any):
        """
        Add the given value to the cache.
        """
        self.cache[cache_key] = value

    def get_code(self, cache_key):
        """
        Get a cached compiled code object.
        """
        return self.code_cache.get(cache_key, None)

    def add_code(self, cache_key, code_obj):
        """
        Cache a compiled code object.
        """
        self.code_cache[cache_key] = code_obj

    def clear(self):
        """
        Clear all cached source, AST, and code objects.
        """
        self.cache.clear()
        self.code_cache.clear()


class Globals:
    """Process-wide tracing state.

    Holds two pieces of true global state:
    - ``saves``: set of ``id()`` for objects marked via ``.save()``.
      The root tracer's ``push()`` filters its frame locals against this
      set so only saved values propagate out of the trace.
    - ``cache``: source/AST/code-object memoization across traces.

    Root-vs-inner detection lives on the tracer itself — see
    ``Tracer.push`` — by checking whether the target frame is an
    nnsight-generated frame (i.e., another trace's compiled body).
    """

    saves = set()

    cache = TracingCache()

    @staticmethod
    def clear():
        Globals.saves.clear()
        Globals.cache.clear()
