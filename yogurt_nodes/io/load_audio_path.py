import hashlib
import os

from .load_audio import load, normalize_time_range, time_range_inputs, trim_waveform


class LoadAudioPath:
    """Load Audio Path node.

    Load audio from path.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "", "tooltip": "Path to the audio or video file"}),
                **time_range_inputs(),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    OUTPUT_NODE = False

    _NODE_NAME = "Load Audio Path"
    DESCRIPTION = "Load audio from path."

    def load_audio(self, audio_path, start_time=0.0, end_time=0.0):
        waveform, sample_rate = load(audio_path)
        waveform = trim_waveform(waveform, sample_rate, start_time, end_time)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

    @classmethod
    def IS_CHANGED(s, audio_path, start_time=0.0, end_time=0.0):
        hasher = hashlib.sha256()
        with open(audio_path, "rb") as audio_file:
            hasher.update(audio_file.read())
        hasher.update(str(normalize_time_range(start_time, end_time)).encode("utf-8"))
        return hasher.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio_path, start_time=0.0, end_time=0.0):
        if not os.path.isfile(audio_path):
            return "Invalid audio file: {}".format(audio_path)
        try:
            normalize_time_range(start_time, end_time)
        except ValueError as exc:
            return str(exc)

        return True
