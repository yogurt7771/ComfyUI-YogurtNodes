import torch


class ReplaceImageInBatch:
    """Replace Image In Batch node.

    Replace one image inside an image batch at the given index.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The image batch to modify."}),
                "image": ("IMAGE", {"tooltip": "Replacement image. If this is a batch, the index will be used when possible; otherwise the first image is used."}),
                "index": ("INT", {"default": 0, "tooltip": "Index to replace. Supports negative indices like Python."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"

    OUTPUT_NODE = False

    _NODE_NAME = "Replace Image In Batch"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Replace one image inside an image batch at the given index."

    def execute(self, images: torch.Tensor, image: torch.Tensor, index: int):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        if images.dim() != 4:
            raise ValueError(f"Expected images as [B,H,W,C], got shape: {tuple(images.shape)}")
        if image.dim() != 4:
            raise ValueError(f"Expected image as [N,H,W,C], got shape: {tuple(image.shape)}")

        batch = int(images.shape[0])
        if batch <= 0:
            return (images,)

        idx = int(index)
        if idx < 0:
            idx += batch
        if idx < 0 or idx >= batch:
            raise ValueError(f"index out of range: {index} (batch={batch})")

        if int(image.shape[0]) == batch:
            replacement = image[idx : idx + 1]
        else:
            replacement = image[0:1]

        if replacement.shape[1:] != images[idx : idx + 1].shape[1:]:
            raise ValueError(
                "Replacement image shape must match batch image shape. "
                f"replacement={tuple(replacement.shape)}, target={tuple(images[idx:idx+1].shape)}"
            )

        out = images.clone()
        out[idx : idx + 1] = replacement.to(device=out.device, dtype=out.dtype)

        return (out,)

