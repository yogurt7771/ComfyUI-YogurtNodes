import os
from pathlib import Path

import folder_paths
from comfy_api.input_impl import VideoFromFile


def list_input_files(content_types):
    input_dir = Path(folder_paths.get_input_directory())
    files = [
        file_path.relative_to(input_dir).as_posix()
        for file_path in input_dir.rglob("*.*")
        if file_path.is_file()
    ]
    return sorted(folder_paths.filter_files_content_types(files, content_types))


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


def get_video_start_and_duration(start_time=0.0, end_time=0.0) -> tuple[float, float]:
    start_time, end_time = normalize_time_range(start_time, end_time)
    duration = end_time - start_time if end_time > 0 else 0.0
    return start_time, duration


class LoadVideo:
    """Load Video node."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": (list_input_files(["video"]), {"video_upload": True}),
                **time_range_inputs(),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load_video"

    OUTPUT_NODE = False

    _NODE_NAME = "Load Video"
    DESCRIPTION = "Load video."

    def load_video(self, video, start_time=0.0, end_time=0.0):
        video_path = folder_paths.get_annotated_filepath(video)
        start_time, duration = get_video_start_and_duration(start_time, end_time)
        return (VideoFromFile(video_path, start_time=start_time, duration=duration),)

    @classmethod
    def IS_CHANGED(s, video, start_time=0.0, end_time=0.0):
        video_path = folder_paths.get_annotated_filepath(video)
        return (os.path.getmtime(video_path), *normalize_time_range(start_time, end_time))

    @classmethod
    def VALIDATE_INPUTS(s, video, start_time=0.0, end_time=0.0):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        try:
            normalize_time_range(start_time, end_time)
        except ValueError as exc:
            return str(exc)

        return True
