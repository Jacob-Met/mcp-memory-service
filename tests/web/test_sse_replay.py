# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SSE Last-Event-ID replay behaviour.

Exercises SSEManager directly with a fake Request so we don't need to
spin up the full FastAPI app or any storage backend.
"""

from dataclasses import dataclass

import pytest

from mcp_memory_service.web.sse import SSEEvent, SSEManager


@dataclass
class _FakeClient:
    host: str = "127.0.0.1"


class _FakeRequest:
    """Minimal fastapi.Request stand-in. Only exposes the attributes the
    SSEManager actually touches: .headers.get(...) and .client.host."""

    def __init__(self, headers=None, client_host: str = "127.0.0.1"):
        self.headers = headers or {}
        self.client = _FakeClient(host=client_host)


async def _drain(queue):
    """Drain a non-empty asyncio.Queue without blocking on the next get."""
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    return drained


@pytest.mark.asyncio
async def test_fresh_connect_has_no_replay_metadata():
    """A connect without Last-Event-ID gets the welcome event only, no replay key."""
    manager = SSEManager(replay_buffer_size=10)
    queue = await manager.add_connection("conn-1", _FakeRequest())

    events = await _drain(queue)
    assert len(events) == 1
    assert events[0].event_type == "connection_established"
    assert "replay" not in events[0].data


@pytest.mark.asyncio
async def test_resume_replays_events_after_id():
    """Reconnect with a known Last-Event-ID replays everything after it."""
    manager = SSEManager(replay_buffer_size=10)

    # Broadcast three events with no live connections (still buffered).
    e1 = SSEEvent(event_type="memory_stored", data={"n": 1})
    e2 = SSEEvent(event_type="memory_stored", data={"n": 2})
    e3 = SSEEvent(event_type="memory_stored", data={"n": 3})
    for e in (e1, e2, e3):
        await manager.broadcast_event(e)

    # Reconnect, claiming we last saw e1.
    queue = await manager.add_connection(
        "conn-r",
        _FakeRequest(headers={"Last-Event-ID": e1.event_id}),
        last_event_id=e1.event_id,
    )
    events = await _drain(queue)

    # Welcome first, then e2, then e3.
    assert events[0].event_type == "connection_established"
    assert events[0].data["replay"]["status"] == "resumed"
    assert events[0].data["replay"]["events_replayed"] == 2
    assert [e.event_id for e in events[1:]] == [e2.event_id, e3.event_id]


@pytest.mark.asyncio
async def test_resume_with_unknown_id_signals_overflow():
    """Reconnect with an id not in the buffer reports id_not_in_buffer and replays nothing."""
    manager = SSEManager(replay_buffer_size=10)
    await manager.broadcast_event(SSEEvent(event_type="memory_stored", data={"n": 1}))

    queue = await manager.add_connection(
        "conn-overflow",
        _FakeRequest(),
        last_event_id="00000000-0000-0000-0000-000000000000",
    )
    events = await _drain(queue)

    assert len(events) == 1  # welcome only
    replay = events[0].data["replay"]
    assert replay["status"] == "id_not_in_buffer"
    assert replay["events_replayed"] == 0


@pytest.mark.asyncio
async def test_filtered_broadcasts_are_not_buffered():
    """Filtered broadcasts target specific live connections; a reconnecting
    client must not receive them via replay because it wasn't an original target."""
    manager = SSEManager(replay_buffer_size=10)

    e_global = SSEEvent(event_type="memory_stored", data={"n": "global"})
    e_filtered = SSEEvent(event_type="memory_stored", data={"n": "filtered"})
    await manager.broadcast_event(e_global)
    await manager.broadcast_event(e_filtered, connection_filter={"some-other-conn"})

    queue = await manager.add_connection(
        "conn-r",
        _FakeRequest(),
        last_event_id=e_global.event_id,
    )
    events = await _drain(queue)

    # Welcome only — no replays after e_global, because e_filtered was not buffered.
    assert len(events) == 1
    assert events[0].data["replay"]["status"] == "resumed"
    assert events[0].data["replay"]["events_replayed"] == 0


@pytest.mark.asyncio
async def test_buffer_evicts_oldest_at_capacity():
    """A small buffer drops the oldest event; resume from an evicted id reports overflow."""
    manager = SSEManager(replay_buffer_size=2)

    e1 = SSEEvent(event_type="memory_stored", data={"n": 1})
    e2 = SSEEvent(event_type="memory_stored", data={"n": 2})
    e3 = SSEEvent(event_type="memory_stored", data={"n": 3})
    for e in (e1, e2, e3):
        await manager.broadcast_event(e)

    # e1 should be evicted; resume from e1 = overflow.
    queue = await manager.add_connection(
        "conn-evicted",
        _FakeRequest(),
        last_event_id=e1.event_id,
    )
    events = await _drain(queue)
    assert events[0].data["replay"]["status"] == "id_not_in_buffer"
    assert events[0].data["replay"]["buffer_oldest_id"] == e2.event_id


@pytest.mark.asyncio
async def test_heartbeats_are_not_buffered():
    """Heartbeats reach live connections but are excluded from the replay
    buffer — otherwise a quiet period would fill the buffer with pings."""
    manager = SSEManager(replay_buffer_size=10)

    real = SSEEvent(event_type="memory_stored", data={"n": 1})
    beat = SSEEvent(event_type="heartbeat", data={"ping": True})
    await manager.broadcast_event(real)
    await manager.broadcast_event(beat)

    queue = await manager.add_connection(
        "conn-hb",
        _FakeRequest(),
        last_event_id=real.event_id,
    )
    events = await _drain(queue)

    # Welcome only — heartbeat was filtered, so nothing to replay after `real`.
    assert len(events) == 1
    assert events[0].data["replay"]["status"] == "resumed"
    assert events[0].data["replay"]["events_replayed"] == 0


@pytest.mark.asyncio
async def test_replay_disabled_when_buffer_size_zero():
    """replay_buffer_size=0 disables replay entirely; Last-Event-ID is ignored."""
    manager = SSEManager(replay_buffer_size=0)
    await manager.broadcast_event(SSEEvent(event_type="memory_stored", data={"n": 1}))

    queue = await manager.add_connection(
        "conn-disabled",
        _FakeRequest(),
        last_event_id="any-id",
    )
    events = await _drain(queue)

    assert len(events) == 1
    assert "replay" not in events[0].data


@pytest.mark.asyncio
async def test_fresh_viewer_joins_an_already_running_consolidation():
    """The late dashboard should not wait for the next phase (which may be minutes away)."""
    manager = SSEManager(replay_buffer_size=0)
    await manager.broadcast_event(SSEEvent(
        event_type='consolidation_started', data={'time_horizon': 'daily'}
    ))
    await manager.broadcast_event(SSEEvent(
        event_type='consolidation_progress',
        data={'time_horizon': 'daily', 'phase': 'clustering', 'completed': 2,
              'total': 4, 'memories_selected': 15},
    ))

    queue = await manager.add_connection('late-viewer', _FakeRequest())
    events = await _drain(queue)
    assert [e.event_type for e in events] == [
        'connection_established', 'consolidation_started', 'consolidation_progress',
    ]
    assert events[-1].data['completed'] == 2

    await manager.broadcast_event(SSEEvent(
        event_type='consolidation_completed',
        data={'time_horizon': 'daily', 'success': True},
    ))
    queue = await manager.add_connection('after-run', _FakeRequest())
    assert [e.event_type for e in await _drain(queue)] == ['connection_established']


@pytest.mark.asyncio
async def test_late_viewer_only_gets_active_consolidations_not_other_memories():
    manager = SSEManager(replay_buffer_size=3)
    await manager.broadcast_event(SSEEvent(event_type='memory_stored', data={'private': 'secret'}))
    for horizon in ('daily', 'weekly'):
        await manager.broadcast_event(SSEEvent(
            event_type='consolidation_started', data={'time_horizon': horizon}
        ))
    queue = await manager.add_connection('scoped-viewer', _FakeRequest(), stores={'team-a'})
    events = await _drain(queue)
    assert [e.event_type for e in events] == [
        'connection_established', 'consolidation_started', 'consolidation_started',
    ]
    assert {e.data['time_horizon'] for e in events[1:]} == {'daily', 'weekly'}


@pytest.mark.asyncio
async def test_filtered_broadcast_does_not_appear_in_late_viewer_snapshot():
    manager = SSEManager(replay_buffer_size=3)
    await manager.broadcast_event(SSEEvent(
        event_type='consolidation_started', data={'time_horizon': 'weekly'}
    ), connection_filter={'another-connection'})
    queue = await manager.add_connection('late', _FakeRequest())
    assert [e.event_type for e in await _drain(queue)] == ['connection_established']


@pytest.mark.asyncio
async def test_late_viewer_gets_latest_measured_phase_without_duplicate_replay():
    manager = SSEManager(replay_buffer_size=10)
    start = SSEEvent(event_type='consolidation_started', data={'time_horizon': 'weekly'})
    first = SSEEvent(event_type='consolidation_progress', data={
        'time_horizon': 'weekly', 'completed': 1, 'total': 4,
    })
    latest = SSEEvent(event_type='consolidation_progress', data={
        'time_horizon': 'weekly', 'completed': 3, 'total': 4,
    })
    for event in (start, first, latest):
        await manager.broadcast_event(event)

    new = await manager.add_connection('new', _FakeRequest())
    assert [event.event_id for event in (await _drain(new))[1:]] == [start.event_id, latest.event_id]

    resumed = await manager.add_connection('resumed', _FakeRequest(), last_event_id=first.event_id)
    events = await _drain(resumed)
    assert [event.event_id for event in events[1:]] == [latest.event_id], 'older start must not erase restored progress'


@pytest.mark.asyncio
async def test_stopped_sse_manager_does_not_report_stale_active_consolidation():
    manager = SSEManager(replay_buffer_size=0)
    await manager.broadcast_event(SSEEvent(
        event_type='consolidation_started', data={'time_horizon': 'daily'},
    ))
    await manager.stop()
    queue = await manager.add_connection('after-restart', _FakeRequest())
    assert [event.event_type for event in await _drain(queue)] == ['connection_established']
