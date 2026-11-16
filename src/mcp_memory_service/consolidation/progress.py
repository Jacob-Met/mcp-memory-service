"""Best-effort dashboard event delivery for all consolidation entry points."""

import logging

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


async def notify_consolidation(event_type: str, data: dict) -> None:
    """Telemetry must never change the result of a consolidation run."""
    try:
        from ..web.sse import SSEEvent, sse_manager

        await sse_manager.broadcast_event(SSEEvent(event_type=event_type, data=data))
    except Exception as exc:
        logger.debug(
            "Unable to notify dashboard of consolidation: %s", _sanitize_log_value(exc)
        )
