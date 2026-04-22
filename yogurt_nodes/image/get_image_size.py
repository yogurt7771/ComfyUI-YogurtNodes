class GetImageSize:

    """Get Image Size node.

    Get image size information.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("image", "batch_size", "width", "height", "channels", "longest_side", "shortest_side", "aspect_ratio", "megapixels",)
    FUNCTION = "execute"

    OUTPUT_NODE = False

    _NODE_NAME = "Get Image Size"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Get image size information."

    def execute(self, image) -> tuple[int, int]:
        batch_size = image.shape[0] if image.dim() >= 1 else 0
        height = image.shape[1] if image.dim() >= 2 else 0
        width = image.shape[2] if image.dim() >= 3 else 0
        channels = image.shape[3] if image.dim() >= 4 else 1

        longest_side = max(width, height)
        shortest_side = min(width, height)

        aspect_ratio = width / height if height != 0 else 0
        megapixels = (width * height) / 1_000_000

        # Send progress text to display size on the node
        return (
            image,
            batch_size,
            width,
            height,
            channels,
            longest_side,
            shortest_side,
            aspect_ratio,
            megapixels,
        )
