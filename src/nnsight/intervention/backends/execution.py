from typing import TYPE_CHECKING, Any

from ..tracing.globals import _mounted_during_execution
from ..tracing.util import wrap_exception
from .base import Backend


if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any


class ExecutionBackend(Backend):

    def __call__(self, tracer: Tracer):

        fn = super().__call__(tracer)

        try:
            # ``Object.save`` is mounted for the duration of execution and
            # unmounted afterwards. ``InterleavingTracer._setup_interleaver``
            # also mounts it, for executors that don't go through this
            # backend (AsyncVLLMBackend, vLLM serve ``server.py``,
            # LocalSimulationBackend).
            with _mounted_during_execution():
                return tracer.execute(fn)
        except Exception as e:

            raise wrap_exception(e, tracer.info) from None
