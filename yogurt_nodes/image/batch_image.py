import torch
import torch.nn.functional as F
import comfy


MAX_RESOLUTION=16384


def resize_image(image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False, pad_value=1):
    _, oh, ow, _ = image.shape
    x = y = x2 = y2 = 0
    pad_left = pad_right = pad_top = pad_bottom = 0

    if keep_proportion:
        method = "keep proportion"

    if multiple_of > 1:
        width = width - (width % multiple_of)
        height = height - (height % multiple_of)

    if method == 'keep proportion' or method == 'pad':
        if width == 0 and oh < height:
            width = MAX_RESOLUTION
        elif width == 0 and oh >= height:
            width = ow

        if height == 0 and ow < width:
            height = MAX_RESOLUTION
        elif height == 0 and ow >= width:
            height = oh

        ratio = min(width / ow, height / oh)
        new_width = round(ow*ratio)
        new_height = round(oh*ratio)

        if method == 'pad':
            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top

        width = new_width
        height = new_height
    elif method.startswith('fill'):
        width = width if width > 0 else ow
        height = height if height > 0 else oh

        ratio = max(width / ow, height / oh)
        new_width = round(ow*ratio)
        new_height = round(oh*ratio)
        x = (new_width - width) // 2
        y = (new_height - height) // 2
        x2 = x + width
        y2 = y + height
        if x2 > new_width:
            x -= (x2 - new_width)
        if x < 0:
            x = 0
        if y2 > new_height:
            y -= (y2 - new_height)
        if y < 0:
            y = 0
        width = new_width
        height = new_height
    else:
        width = width if width > 0 else ow
        height = height if height > 0 else oh

    if "always" in condition \
        or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
        or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

        outputs = image.permute(0,3,1,2)

        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        if method == 'pad':
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

        outputs = outputs.permute(0,2,3,1)

        if method.startswith('fill'):
            if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                outputs = outputs[:, y:y2, x:x2, :]
    else:
        outputs = image

    if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
        width = outputs.shape[2]
        height = outputs.shape[1]
        x = (width % multiple_of) // 2
        y = (height % multiple_of) // 2
        x2 = width - ((width % multiple_of) - x)
        y2 = height - ((height % multiple_of) - y)
        outputs = outputs[:, y:y2, x:x2, :]
    
    outputs = torch.clamp(outputs, 0, 1)

    return outputs


class BatchImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "interpolation": (
                    [
                        "nearest",
                        "bilinear",
                        "bicubic",
                        "area",
                        "nearest-exact",
                        "lanczos",
                    ],
                    {"default": "nearest"},
                ),
                "method": (["stretch", "fill / crop", "pad"], {"default": "pad"}),
                "pad_value": ("FLOAT", {"default": 1.0}),
                "start_index": ("INT", {"default": 0, "tooltip": "The start index. Same as Python slicing."}),
                "end_index": ("INT", {"default": 0, "tooltip": "The end index. Same as Python slicing. 0 means the end. Negative values are also supported."}),
                "step": ("INT", {"default": 1, "tooltip": "The step. Same as Python slicing."}),
            },
            "optional": {
                "images1": ("IMAGE", {"tooltip": "The images to batch."}),
                "images2": ("IMAGE", {"tooltip": "The images to batch."}),
                "images3": ("IMAGE", {"tooltip": "The images to batch."}),
                "images4": ("IMAGE", {"tooltip": "The images to batch."}),
                "images5": ("IMAGE", {"tooltip": "The images to batch."}),
                "images6": ("IMAGE", {"tooltip": "The images to batch."}),
                "images7": ("IMAGE", {"tooltip": "The images to batch."}),
                "images8": ("IMAGE", {"tooltip": "The images to batch."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "batch_image"

    OUTPUT_NODE = False

    _NODE_NAME = "Batch Images"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Batch images."

    def batch_image(
        self,
        images1: torch.Tensor = None,
        images2: torch.Tensor = None,
        images3: torch.Tensor = None,
        images4: torch.Tensor = None,
        images5: torch.Tensor = None,
        images6: torch.Tensor = None,
        images7: torch.Tensor = None,
        images8: torch.Tensor = None,
        interpolation="nearest",
        method="pad",
        pad_value=1.0,
        start_index=0,
        end_index=0,
        step=1,
    ):
        """
        Batch images.
        """
        images = [images1, images2, images3, images4, images5, images6, images7, images8]
        images = [image for image in images if image is not None]
        if len(images) == 0:
            return (None,)
        new_height = max([image.shape[1] for image in images])
        new_width = max([image.shape[2] for image in images])
        for i in range(len(images)):
            if new_height != images[i].shape[1] or new_width != images[i].shape[2]:
                images[i] = resize_image(images[i], new_width, new_height, method=method, interpolation=interpolation, condition="always", multiple_of=0, keep_proportion=False, pad_value=pad_value)
        batch = (torch.cat(images, dim=0),)
        if end_index == 0:
            batch = batch[start_index::step]
        else:
            batch = batch[start_index:end_index:step]
        if len(batch) == 0:
            return (None,)
        return batch
