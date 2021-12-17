# voice activity detection
import numpy as np
import threading, queue
import pyaudio
import torch


class VAD:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000
    # CHUNK = int(SAMPLE_RATE / 10)
    CHUNK = 1024
    FRAME_DURATION_MS = 250

    def __init__(self, confidence=0.85):
        self.confidence = confidence
        self.audio = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.thread = None
        self.running = False
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        self.model = model

    def get_voice(self, block=True):
        return self.queue.get(block)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._recording)
            self.thread.start()

    def stop(self):
        self.running = False
        self.queue.put(b"")
        self.thread.join()
        self.thread = None

    def _recording(self):
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        data = []

        print(f"sample_size: {self.audio.get_sample_size(self.FORMAT)}")

        self.running = True
        while self.running:

            audio_chunk = stream.read(
                int(self.SAMPLE_RATE * self.FRAME_DURATION_MS / 1000.0)
            )

            audio_int16 = np.frombuffer(audio_chunk, np.int16)

            audio_float32 = self._int2float(audio_int16)

            # get the confidences and add them to the list to plot them later
            vad_outs = self._validate(torch.from_numpy(audio_float32))

            # get the confidence value so that jupyterplot can process it
            confidence = vad_outs[:, 1].numpy()[0].item()
            if confidence > self.confidence:
                data.append(audio_chunk)
                print(f"speeking {confidence:.2f}")
            else:
                if len(data):
                    self.queue.put(b"".join(data))
                    data = []

    def _int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype("float32")
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def _validate(self, inputs: torch.Tensor):
        with torch.no_grad():
            outs = self.model(inputs)
        return outs


if __name__ == "__main__":
    vad = VAD()
    vad.start()
    try:
        while True:
            vad.get_voice()
    except KeyboardInterrupt:
        print("stop")
        vad.stop()
