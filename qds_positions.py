"""
Timestamped quDIS interferometer positions for the Object Accumulator.

The interferometer is read on a Windows machine and streamed over ZMQ by
ptychoSAXS/qds_bridge/qds_publisher.py. Every sample carries the publisher's
wall-clock time in Unix-epoch seconds — the same epoch as the detector PV's
timeStamp — so a frame can be paired with the sample taken closest to it.

QDSHistory subscribes in a background thread and keeps the most recent
samples in a bounded deque; closest() does the pairing.

The wire format and host/port come from the sibling ptychoSAXS checkout, so
they cannot drift from what the publisher sends. Its wire.py and config.py
are loaded by file path rather than by putting ptychoSAXS on sys.path, whose
top-level packages (gui, tools, tests, ...) would otherwise be importable
from this process. If either file or pyzmq is missing, QDS_AVAILABLE is False
and QDS_UNAVAILABLE_REASON says why; nothing else here is usable then.

No Qt in this module, so it can be exercised on its own.
"""

import bisect
import importlib.util
import struct
import threading
import time
from collections import deque
from pathlib import Path

_BRIDGE_DIR = Path(__file__).resolve().parent.parent / 'ptychoSAXS' / 'qds_bridge'

# Horizontal and vertical are the second and third interferometer channels.
_H_CHANNEL = 1
_V_CHANNEL = 2

_POLL_TIMEOUT_MS = 500

DEFAULT_MAXLEN = 100
DEFAULT_EVERY_N = 1


def _load(name):
    path = _BRIDGE_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'_qds_bridge_{name}', path)
    if spec is None:
        raise ImportError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    import zmq
    _wire = _load('wire')
    _config = _load('config')
    if not hasattr(_wire, 'unpack_sample_ts'):
        raise ImportError(f'{_BRIDGE_DIR / "wire.py"} predates sample '
                          f'timestamps; update ptychoSAXS')
    QDS_AVAILABLE = True
    QDS_UNAVAILABLE_REASON = ''
    DEFAULT_HOST = _config.QDS_HOST
    DEFAULT_PORT = _config.QDS_PORT
except Exception as exc:  # noqa: BLE001 - any failure just disables the source
    zmq = _wire = _config = None
    QDS_AVAILABLE = False
    QDS_UNAVAILABLE_REASON = (f'QDS bridge unavailable ({type(exc).__name__}: '
                              f'{exc}). Needs pyzmq and ptychoSAXS checked '
                              f'out at {_BRIDGE_DIR.parent}.')
    DEFAULT_HOST = 'sec12mic'
    DEFAULT_PORT = 5561

PUBLISHER_HINT = (
    'Is qds_publisher.py running? As of Oct 2026: log in to s12ide@sec12mic '
    'and double-click qds_publisher on the desktop, or in the Anaconda '
    'terminal run "conda activate ptychoSAXS", cd to '
    'C:\\Users\\s12ide\\Documents\\GitHub\\ptychoSAXS\\qds_bridge and run '
    '"python qds_publisher.py".'
)


class QDSHistory:
    """The last `maxlen` timestamped (t, h_nm, v_nm) samples from the bridge.

    Only every `every_n`-th received sample is kept, which stretches the time
    the deque covers without making it longer. Samples from a publisher that
    sends no timestamps are counted but never stored, since they cannot be
    paired with anything.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 maxlen=DEFAULT_MAXLEN, every_n=DEFAULT_EVERY_N):
        if not QDS_AVAILABLE:
            raise RuntimeError(QDS_UNAVAILABLE_REASON)
        self.endpoint = _config.endpoint(host, port)
        self._lock = threading.Lock()
        self._samples = deque(maxlen=max(2, int(maxlen)))
        self._every_n = max(1, int(every_n))
        self._skip = 0
        self.received = 0
        self.untimed = 0
        self.malformed = 0
        self.connect_error = None
        self._first = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='qds-history',
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def wait_first(self, timeout):
        """True once a timestamped sample is in the deque."""
        return self._first.wait(timeout)

    # -- settings, safe to change while running -----------------------------

    @property
    def maxlen(self):
        return self._samples.maxlen

    @property
    def every_n(self):
        return self._every_n

    def set_maxlen(self, n):
        with self._lock:
            self._samples = deque(self._samples, maxlen=max(2, int(n)))

    def set_every_n(self, n):
        with self._lock:
            self._every_n = max(1, int(n))
            self._skip = 0

    # -- reading ------------------------------------------------------------

    def newest(self):
        """The most recent (t, h_nm, v_nm), or None if nothing is stored."""
        with self._lock:
            return self._samples[-1] if self._samples else None

    def closest(self, t_target):
        """The stored sample nearest in time to `t_target`.

        Returns (t, h_nm, v_nm, is_oldest, n_stored), or None if the deque is
        empty. `is_oldest` means the match is the oldest sample kept, so the
        true nearest may already have been pushed out of the deque.

        Several samples can share a timestamp: before Python 3.13, the
        publisher's time.time() on Windows only advances every ~15.6 ms. The
        earliest of such a group is used, and an exact tie in distance also
        goes to the earlier sample.
        """
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return None
        times = [s[0] for s in samples]
        i = bisect.bisect_left(times, t_target)
        if i == len(samples):
            i -= 1
        elif i > 0 and t_target - times[i - 1] <= times[i] - t_target:
            i -= 1
        while i > 0 and times[i - 1] == times[i]:
            i -= 1
        t, h, v = samples[i]
        return t, h, v, i == 0, len(samples)

    def stats(self):
        """(stored, oldest_t, newest_t) for diagnostics."""
        with self._lock:
            if not self._samples:
                return 0, None, None
            return len(self._samples), self._samples[0][0], self._samples[-1][0]

    # -- receive thread -----------------------------------------------------

    def _run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        try:
            sock.setsockopt(zmq.RCVHWM, _config.RCV_HWM)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SUBSCRIBE, _config.TOPIC)
            sock.connect(self.endpoint)
        except zmq.ZMQError as exc:
            self.connect_error = str(exc)
            sock.close()
            ctx.term()
            return

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not poller.poll(timeout=_POLL_TIMEOUT_MS):
                    continue
                while True:
                    try:
                        frames = sock.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if len(frames) != 2:
                        self.malformed += 1
                        continue
                    try:
                        sample = _wire.unpack_sample_ts(frames[1])
                    except struct.error:
                        self.malformed += 1
                        continue
                    self.received += 1
                    t = sample[4]
                    if t is None:
                        self.untimed += 1
                        continue
                    with self._lock:
                        if self._skip:
                            self._skip -= 1
                            continue
                        self._skip = self._every_n - 1
                        self._samples.append(
                            (t, sample[1 + _H_CHANNEL], sample[1 + _V_CHANNEL]))
                    self._first.set()
        finally:
            sock.close()
            ctx.term()


def age_s(t):
    """Seconds since publisher time `t`, by this machine's clock."""
    return time.time() - t
