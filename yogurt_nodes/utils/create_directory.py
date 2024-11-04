from pathlib import Path


class CreateDirectory:
    """
    Create a directory
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"placeholder": "Input path"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_directory"
    DESCRIPTION = "Create a directory"
    CATEGORY = "YogurtNodes/Utils"
    NODE_NAME = "Create Directory"

    def create_directory(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return (path,)
