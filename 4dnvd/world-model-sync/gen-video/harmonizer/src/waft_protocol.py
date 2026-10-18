# SPDX-License-Identifier: Apache-2.0
"""Length-prefixed, pickle-free NumPy messages across the two Pixi environments."""

import io
import os
import select
import struct
import time

import numpy as np


def send_message(stream, **arrays):
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    data = payload.getvalue()
    for part in (struct.pack("!Q", len(data)), data):
        remaining = memoryview(part)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise BrokenPipeError("WAFT message pipe is closed")
            remaining = remaining[written:]
    stream.flush()


def receive_message(stream, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout

    def read_exact(size):
        parts = []
        while size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                    raise TimeoutError("Timed out waiting for the WAFT worker")
                chunk = os.read(stream.fileno(), min(size, 1024 * 1024))
            else:
                chunk = stream.read(size)
            if not chunk:
                raise EOFError("WAFT worker closed its output pipe; check its stderr")
            parts.append(chunk)
            size -= len(chunk)
        return b"".join(parts)

    size = struct.unpack("!Q", read_exact(8))[0]
    if size > 1024 * 1024 * 1024:
        raise ValueError("Invalid WAFT message size")
    with np.load(io.BytesIO(read_exact(size)), allow_pickle=False) as message:
        return {key: message[key] for key in message.files}
