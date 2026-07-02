"""
eeg_stream.py  —  NöroART EEG Streaming Module
================================================
Uses the official Emotiv Cortex SDK (cortex-example-master/python/cortex.py)
to stream EEG data from an Emotiv headset.

Public interface (unchanged from the original version):
    stream = EEGStream(client_id, client_secret)
    stream.connect()              # blocks until EEG is ready
    window = stream.get_window()  # returns np.ndarray shape (128, 14)
"""

import sys
import os
import ssl
import threading
import queue
import time
from datetime import datetime

import numpy as np

# ── Add official Cortex SDK to Python path ───────────────────────────────────
_SDK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cortex-example-master", "python")
if _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

import websocket          # websocket-client (already required by Cortex SDK)
from cortex import Cortex # official Emotiv SDK

# ── Standard EPOC 14-channel names (in headset order) ────────────────────────
EPOC_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2",  "P8", "T8", "FC6", "F4", "F8", "AF4",
]


# ── Cortex subclass: skip SSL cert check (matches original behaviour) ─────────
class _CortexLocal(Cortex):
    """
    Thin Cortex subclass that replaces the upstream SSL certificate path
    (relative '../certificates/rootCA.pem') with CERT_NONE, identical to
    the original hand-written eeg_stream.py.
    """
    def open(self):
        url = "wss://localhost:6868"
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self.on_message,
            on_open=self.on_open,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        thread_name = "CortexWS-{:%Y%m%d%H%M%S}".format(datetime.now())
        sslopt = {"cert_reqs": ssl.CERT_NONE}
        self.websock_thread = threading.Thread(
            target=self.ws.run_forever,
            args=(None, sslopt),
            name=thread_name,
            daemon=True,
        )
        self.websock_thread.start()
        self.websock_thread.join()   # blocks until WebSocket closes


# ── Main EEGStream class ──────────────────────────────────────────────────────
class EEGStream:
    """
    High-level EEG streaming interface for NöroART.

    Wraps the official Emotiv Cortex SDK and exposes the same simple
    synchronous API as the original eeg_stream.py so app.py is unchanged.

    Workflow handled internally
    ---------------------------
    check_access_right → authorize → connect headset → create session
    → subscribe EEG → buffer samples → get_window()
    """

    def __init__(self, client_id: str, client_secret: str,
                 headset_id: str = "", debug: bool = False):

        self.client_id     = client_id
        self.client_secret = client_secret
        self.headset_id    = headset_id

        # Thread-safe sample buffer
        self._sample_queue   = queue.Queue()
        self._buffer         = []
        self._eeg_ch_indices = []   # populated once labels arrive
        self.eeg_labels      = None

        # Synchronisation
        self._session_ready = threading.Event()
        self._error         = None

        # Cortex client
        self.c = _CortexLocal(client_id, client_secret, debug_mode=debug)
        self._bind_events()

    # ── Event binding ─────────────────────────────────────────────────────────

    def _bind_events(self):
        self.c.bind(create_session_done=self._on_session_done)
        self.c.bind(new_data_labels=self._on_data_labels)
        self.c.bind(new_eeg_data=self._on_new_eeg_data)
        self.c.bind(inform_error=self._on_error)
        self.c.bind(warn_cortex_stop_all_sub=self._on_stop)

    # ── Public: connect / disconnect ──────────────────────────────────────────

    def connect(self, timeout: int = 60):
        """
        Connect to Emotiv Cortex in a background daemon thread and block
        until the EEG subscription is confirmed (or timeout is reached).

        Parameters
        ----------
        timeout : int
            Maximum seconds to wait.  Default 60.
        """
        print("🔌 Connecting to Emotiv Cortex (official SDK)…")

        if self.headset_id:
            self.c.set_wanted_headset(self.headset_id)

        # Cortex.open() blocks internally (join on websocket thread),
        # so we run it in a separate daemon thread.
        self._cortex_thread = threading.Thread(
            target=self.c.open,
            daemon=True,
            name="CortexThread",
        )
        self._cortex_thread.start()

        print(f"⏳ Waiting for EEG session (timeout={timeout}s)…")
        ready = self._session_ready.wait(timeout=timeout)

        if not ready:
            raise TimeoutError("❌ Cortex session not established within timeout.")
        if self._error:
            raise RuntimeError(f"❌ Cortex error: {self._error}")

        print("✅ EEG stream ready!")

    def disconnect(self):
        """Gracefully close the Cortex session and WebSocket."""
        print("🔌 Disconnecting from Cortex…")
        try:
            self.c.close_session()
            time.sleep(0.5)
            self.c.close()
        except Exception as exc:
            print(f"⚠️  Disconnect error: {exc}")

    # ── Public: baseline recording ──────────────────────────────────────────

    def record_baseline(self, duration_sec: int = 30) -> np.ndarray:
        """
        Collect `duration_sec` seconds of resting-state EEG and return
        the per-channel mean — ready to subtract from future windows.

        Parameters
        ----------
        duration_sec : int
            How many seconds of resting signal to record.  Default 30.

        Returns
        -------
        np.ndarray, shape (14,), dtype float32
            Mean amplitude per EEG channel over the baseline period.
        """
        n_samples = duration_sec * 128          # 128 Hz
        print(f"[Baseline] Recording {duration_sec}s resting EEG ({n_samples} samples)…")

        collected = []
        while len(collected) < n_samples:
            try:
                sample = self._sample_queue.get(timeout=5.0)
                collected.append(sample)
                done = len(collected)
                if done % 128 == 0:            # print progress every second
                    print(f"[Baseline] {done // 128}/{duration_sec}s collected")
            except queue.Empty:
                print("[Baseline] WARNING: no EEG data — retrying…")
                time.sleep(0.1)

        baseline_array = np.array(collected, dtype=np.float32)   # (N, 14)
        baseline_mean  = baseline_array.mean(axis=0)              # (14,)
        print(f"[Baseline] Done. Per-channel mean: {baseline_mean.round(2)}")
        return baseline_mean

    # ── Public: get EEG window ────────────────────────────────────────────────

    def get_window(self, window_size: int = 128) -> np.ndarray:
        """
        Collect `window_size` EEG samples and return as a NumPy array.

        At 128 Hz, window_size=128 corresponds to 1 second of data.

        Returns
        -------
        np.ndarray, shape (window_size, 14), dtype float32
            14 channels in EPOC order:
            AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        """
        while len(self._buffer) < window_size:
            try:
                sample = self._sample_queue.get(timeout=5.0)
                self._buffer.append(sample)
            except queue.Empty:
                print("⚠️  EEG timeout — waiting for data…")
                time.sleep(0.1)

        window = np.array(self._buffer[:window_size], dtype=np.float32)
        self._buffer = self._buffer[window_size:]
        return window

    # ── Private Cortex event callbacks ────────────────────────────────────────

    def _on_session_done(self, *args, **kwargs):
        session_id = kwargs.get("data", "unknown")
        print(f"✅ Cortex session created: {session_id}")
        # Subscribe to raw EEG stream
        self.c.sub_request(["eeg"])

    def _on_data_labels(self, *args, **kwargs):
        """
        Called once the EEG subscription is confirmed.
        Resolves channel indices from the label list, then unblocks connect().
        """
        data = kwargs.get("data", {})
        if data.get("streamName") != "eeg":
            return

        self.eeg_labels = data.get("labels", [])
        print(f"📊 EEG labels received: {self.eeg_labels}")

        # Map EPOC channel names → indices in the incoming data array
        self._eeg_ch_indices = [
            self.eeg_labels.index(ch)
            for ch in EPOC_CHANNELS
            if ch in self.eeg_labels
        ]
        print(f"📊 Channel indices: {self._eeg_ch_indices}")

        if not self._eeg_ch_indices:
            self._error = "No recognised EPOC channels in EEG labels."

        self._session_ready.set()   # unblock connect()

    def _on_new_eeg_data(self, *args, **kwargs):
        """
        Called ~128×/sec with each EEG packet.
        Cortex SDK already removes the MARKER from the end of the array.
        """
        data = kwargs.get("data")
        if not data or "eeg" not in data or not self._eeg_ch_indices:
            return
        raw = data["eeg"]
        try:
            sample = [raw[i] for i in self._eeg_ch_indices]
            self._sample_queue.put(sample)
        except IndexError:
            pass    # malformed packet — skip silently

    def _on_error(self, *args, **kwargs):
        error_data = kwargs.get("error_data", {})
        self._error = error_data.get("message", str(error_data))
        print(f"⚠️  Cortex error: {self._error}")
        self._session_ready.set()   # unblock connect() so it can raise

    def _on_stop(self, *args, **kwargs):
        print("⚠️  Cortex stopped all subscriptions.")