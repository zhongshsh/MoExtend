import torch
import torch.nn as nn
import os
import re
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


def build_vision_projector(mm_hidden_size=1024, hidden_size=4096):
    projector_type = "mlp2x_gelu"

    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    raise ValueError(f"Unknown projector type: {projector_type}")


class CLIPVisionTower(nn.Module):
    def __init__(
        self, vision_tower="openai/clip-vit-large-patch14-336", delay_load=True
    ):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.select_feature = "patch"

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        self.mm_projector = build_vision_projector()

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            try:
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                )  # error NotImplementedError: Cannot copy out of meta tensor; no data!
            except:
                image_forward_outs = self.vision_tower(
                    images.to(dtype=self.dtype), output_hidden_states=True
                )

            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        image_features = image_features.to(
            next(self.mm_projector.parameters()).dtype
        ).detach()
        image_features = self.mm_projector(image_features)
        return image_features

    def encode(self, image_paths):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(
                self.image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            )
        images = torch.stack(images, dim=0)
        image_features = self(images)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
