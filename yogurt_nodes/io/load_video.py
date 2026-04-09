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


class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": (list_input_files(["video"]), {"video_upload": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load_video"

    OUTPUT_NODE = False

    _NODE_NAME = "Load Video"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Load video."

    def load_video(self, video):
        video_path = folder_paths.get_annotated_filepath(video)
        return (VideoFromFile(video_path),)

    @classmethod
    def IS_CHANGED(s, video):
        video_path = folder_paths.get_annotated_filepath(video)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)

        return True
