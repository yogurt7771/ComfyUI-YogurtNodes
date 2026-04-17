import torch
import torchvision


def pil_image_to_tensor(image):
    return torchvision.transforms.ToTensor()(image).permute(1, 2, 0).unsqueeze(0)


def empty_image_tensor():
    return torch.zeros(1, 1, 1, 3, dtype=torch.float32)


def build_image_outputs(images):
    if not images:
        return empty_image_tensor(), [], 0

    image_list = [pil_image_to_tensor(image) for image in images]
    return image_list[0], image_list, len(image_list)
