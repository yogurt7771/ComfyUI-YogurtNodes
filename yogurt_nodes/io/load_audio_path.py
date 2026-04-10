import hashlib
import os

import av
import torch


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def load(filepath: str) -> tuple[torch.Tensor, int]:
    with av.open(filepath) as audio_file:
        if not audio_file.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = audio_file.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        channel_count = stream.channels

        frames = []
        for frame in audio_file.decode(streams=stream.index):
            buffer = torch.from_numpy(frame.to_ndarray())
            if buffer.shape[0] != channel_count:
                buffer = buffer.view(-1, channel_count).t()

            frames.append(buffer)

        if not frames:
            raise ValueError("No audio frames decoded.")

        waveform = torch.cat(frames, dim=1)
        return f32_pcm(waveform), sample_rate


def time_range_inputs():
    return {
        "start_time": (
            "FLOAT,INT",
            {
                "default": 0.0,
                "min": 0.0,
                "step": 0.01,
                "widgetType": "FLOAT",
                "tooltip": "Start time in seconds. 0 means from the beginning.",
            },
        ),
        "end_time": (
            "FLOAT,INT",
            {
                "default": 0.0,
                "min": 0.0,
                "step": 0.01,
                "widgetType": "FLOAT",
                "tooltip": "End time in seconds. 0 means no end limit.",
            },
        ),
    }


def normalize_time_value(value) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def normalize_time_range(start_time=0.0, end_time=0.0) -> tuple[float, float]:
    start_time = normalize_time_value(start_time)
    end_time = normalize_time_value(end_time)

    if start_time < 0:
        raise ValueError("start_time must be greater than or equal to 0.")
    if end_time < 0:
        raise ValueError("end_time must be greater than or equal to 0.")
    if end_time > 0 and end_time <= start_time:
        raise ValueError("end_time must be greater than start_time, or 0 to disable the end limit.")

    return start_time, end_time


def trim_waveform(
    waveform: torch.Tensor, sample_rate: int, start_time=0.0, end_time=0.0
) -> torch.Tensor:
    start_time, end_time = normalize_time_range(start_time, end_time)

    start_frame = min(int(start_time * sample_rate), waveform.shape[-1])
    end_frame = int(end_time * sample_rate) if end_time > 0 else waveform.shape[-1]
    end_frame = min(end_frame, waveform.shape[-1])

    return waveform[..., start_frame:end_frame]


class LoadAudioPath:
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
    CATEGORY = "YogurtNodes/IO"
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
