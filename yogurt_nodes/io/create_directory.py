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
    RETURN_NAMES = ("path",)

    FUNCTION = "create_directory"

    _NODE_NAME = "Create Directory"
    DESCRIPTION = "Create a directory"
    CATEGORY = "YogurtNodes/IO"

    def create_directory(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return (path,)


class CreateParentDirectory:
    """
    Create a parent directory
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"placeholder": "Input path"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)

    FUNCTION = "create_parent_directory"

    _NODE_NAME = "Create Parent Directory"
    DESCRIPTION = "Create a parent directory"
    CATEGORY = "YogurtNodes/IO"

    def create_parent_directory(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return (path,)
