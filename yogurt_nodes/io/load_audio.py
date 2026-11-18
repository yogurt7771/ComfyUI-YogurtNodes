import hashlib
import os
from fractions import Fraction

import av
import torch

import folder_paths

from .input_files import list_input_files


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def load(filepath: str) -> tuple[torch.Tensor, int]:
    """Decode the first audio stream to a float (channels, samples) tensor.

    Packets the decoder rejects as invalid data (including the end-of-stream
    flush) are skipped and replaced by silence covering the gap (from the next
    frame's pts, else the packets' durations), so the timeline stays aligned. Frames whose sample format, channel layout or rate
    differ from the first frame are converted to match it.
    """
    with av.open(filepath) as audio_file:
        if not audio_file.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = audio_file.streams.audio[0]
        items, skipped, first_error = _decode_packets(audio_file, stream)

        if not any(isinstance(item, av.AudioFrame) for item in items):
            message = "No audio frames decoded."
            if first_error is not None:
                message += f" All {skipped} packet(s) failed to decode: {first_error}"
            raise ValueError(message)

        waveform, sample_rate = _concat_frames(items)
        if skipped:
            print(
                f"[YogurtNodes] Load Audio: skipped {skipped} undecodable packet(s) in "
                f"{os.path.basename(filepath)} ({first_error}); filled the gaps with silence."
            )
        return waveform, sample_rate


def _decode_packets(audio_file, stream):
    """Return ([AudioFrame | float silence_seconds, ...], skipped_count, first_error)."""
    items = []
    skipped = 0
    first_error = None
    expected_time = None  # where the next frame should start, from the last good frame's pts
    lost_time = None  # summed duration of packets skipped since the last good frame

    # demux() ends with an empty packet per stream; decoding it flushes the decoder.
    for packet in audio_file.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as exc:
            skipped += 1
            first_error = first_error or exc
            lost_time = (lost_time or 0) + (
                packet.duration * packet.time_base if packet.duration and packet.time_base else 0
            )
            continue

        for frame in frames:
            frame_time = frame.pts * frame.time_base if frame.pts is not None else None
            if lost_time is not None:
                # Prefer the pts jump: some parsers (e.g. FLAC) emit damaged packets without duration.
                if frame_time is not None and expected_time is not None:
                    lost_time = frame_time - expected_time
                if lost_time > 0:
                    items.append(float(lost_time))
                lost_time = None
            items.append(frame)
            if frame_time is not None:
                expected_time = frame_time + Fraction(frame.samples, frame.sample_rate)

    if lost_time:
        items.append(float(lost_time))
    return items, skipped, first_error


def _concat_frames(items) -> tuple[torch.Tensor, int]:
    reference = next(item for item in items if isinstance(item, av.AudioFrame))
    target = (reference.format.name, reference.layout.name, reference.sample_rate)

    chunks = []
    resampler, resampler_source = None, None

    def flush_resampler():
        if resampler is not None:
            chunks.extend(_frame_tensor(frame) for frame in resampler.resample(None))

    for item in items:
        if not isinstance(item, av.AudioFrame):
            chunks.append(item)
            continue
        source = (item.format.name, item.layout.name, item.sample_rate)
        if source == target:
            flush_resampler()
            resampler, resampler_source = None, None
            chunks.append(_frame_tensor(item))
            continue
        if source != resampler_source:
            flush_resampler()
            resampler = av.AudioResampler(format=target[0], layout=target[1], rate=target[2])
            resampler_source = source
        chunks.extend(_frame_tensor(frame) for frame in resampler.resample(item))
    flush_resampler()

    first = next(chunk for chunk in chunks if isinstance(chunk, torch.Tensor))
    tensors = [
        chunk
        if isinstance(chunk, torch.Tensor)
        else first.new_zeros((first.shape[0], round(chunk * target[2])))
        for chunk in chunks
    ]
    return torch.cat(tensors, dim=1), reference.sample_rate


def _frame_tensor(frame: av.AudioFrame) -> torch.Tensor:
    buffer = torch.from_numpy(frame.to_ndarray())
    if not frame.format.is_planar:
        buffer = buffer.view(-1, len(frame.layout.channels)).t()
    return f32_pcm(buffer)


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




class LoadAudio__V3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": (list_input_files(["audio", "video"]), {"audio_upload": True}),
                **time_range_inputs(),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"

    OUTPUT_NODE = False

    _NODE_NAME = "Load Audio"
    DESCRIPTION = "Load audio."

    def load_audio(self, audio, start_time=0.0, end_time=0.0):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load(audio_path)
        waveform = trim_waveform(waveform, sample_rate, start_time, end_time)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

    @classmethod
    def IS_CHANGED(s, audio, start_time=0.0, end_time=0.0):
        audio_path = folder_paths.get_annotated_filepath(audio)
        hasher = hashlib.sha256()
        with open(audio_path, "rb") as audio_file:
            hasher.update(audio_file.read())
        hasher.update(str(normalize_time_range(start_time, end_time)).encode("utf-8"))
        return hasher.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio, start_time=0.0, end_time=0.0):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        try:
            normalize_time_range(start_time, end_time)
        except ValueError as exc:
            return str(exc)

        return True


LoadAudio = LoadAudio__V3
