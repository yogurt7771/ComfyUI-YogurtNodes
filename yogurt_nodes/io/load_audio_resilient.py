from pathlib import Path

import av
import torch

import folder_paths

from .load_audio import LoadAudio__V3, f32_pcm, trim_waveform


def load_resilient(filepath: str) -> tuple[torch.Tensor, int]:
    with av.open(filepath) as audio_file:
        if not audio_file.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = audio_file.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        channel_count = stream.channels
        frames = []
        skipped_packets = 0

        for packet in audio_file.demux(stream):
            try:
                decoded_frames = packet.decode()
            except av.error.InvalidDataError:
                skipped_packets += 1
                continue

            for frame in decoded_frames:
                buffer = torch.from_numpy(frame.to_ndarray())
                if buffer.shape[0] != channel_count:
                    buffer = buffer.view(-1, channel_count).t()
                frames.append(buffer)

        if not frames:
            raise ValueError("No audio frames decoded.")
        if skipped_packets:
            print(f"[Yogurt] Skipped {skipped_packets} damaged audio packet(s) in {Path(filepath).name}.")

        waveform = torch.cat(frames, dim=1)
        return f32_pcm(waveform), sample_rate


class LoadAudioResilient__V3(LoadAudio__V3):
    _NODE_NAME = "Load Audio (Resilient)"
    DESCRIPTION = "Load audio from input folders, skipping isolated damaged packets."

    def load_audio(self, audio, start_time=0.0, end_time=0.0):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load_resilient(audio_path)
        waveform = trim_waveform(waveform, sample_rate, start_time, end_time)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


LoadAudioResilient = LoadAudioResilient__V3
