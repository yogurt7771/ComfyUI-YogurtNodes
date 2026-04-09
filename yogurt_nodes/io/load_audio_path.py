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


class LoadAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "", "tooltip": "Path to the audio or video file"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    OUTPUT_NODE = False

    _NODE_NAME = "Load Audio Path"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Load audio from path."

    def load_audio(self, audio_path):
        waveform, sample_rate = load(audio_path)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

    @classmethod
    def IS_CHANGED(s, audio_path):
        hasher = hashlib.sha256()
        with open(audio_path, "rb") as audio_file:
            hasher.update(audio_file.read())
        return hasher.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio_path):
        if not os.path.isfile(audio_path):
            return "Invalid audio file: {}".format(audio_path)

        return True
