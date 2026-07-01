import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetModel


def get_resnet_model(device):
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
    model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
    model.eval()
    return model, processor


def get_resnet_embeddings(processor, model, device, pil_images: list[Image.Image]):
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.cpu().numpy().squeeze()
