import os

from comfy_api.input_impl import VideoFromFile


class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "tooltip": "Path to the video file"}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load_video"

    OUTPUT_NODE = False

    _NODE_NAME = "Load Video Path"
    CATEGORY = "YogurtNodes/IO"
    DESCRIPTION = "Load video from path."

    def load_video(self, video_path):
        return (VideoFromFile(video_path),)

    @classmethod
    def IS_CHANGED(s, video_path):
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(s, video_path):
        if not os.path.isfile(video_path):
            return "Invalid video file: {}".format(video_path)

        return True
