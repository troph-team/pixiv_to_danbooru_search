import torch
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-large-patch14"


class CLIPEmbedder:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(
            self.device, dtype=self.dtype)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def __encode_image(self, pixel_values):
        feature = self.model.get_image_features(pixel_values=pixel_values)
        feature /= feature.norm(dim=-1, keepdim=True)
        return feature

    @torch.no_grad()
    def __call__(self, images):
        image_tensor = self.processor(images=images, return_tensors="pt").to(
            self.device
        )
        image_features = self.__encode_image(image_tensor["pixel_values"])
        image_emb = image_features.cpu().to(torch.float16).detach().numpy()
        return image_emb


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import os

    img = Image.open("/home/ubuntu/gradio_tool/87195846_p0_resized.webp")

    embedder = CLIPEmbedder(device="cpu")
    emb = embedder([img])
    np.save("image_1.npy", emb)