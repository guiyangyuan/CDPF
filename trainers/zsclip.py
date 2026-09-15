import torch

from dassl.engine import TRAINER_REGISTRY, TrainerX

from clip import clip

from .coop import CUSTOM_TEMPLATES, load_clip_to_cpu


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def load_model(self, directory, epoch=None):
        if directory:
            raise ValueError("ZeroshotCLIP does not load a learned checkpoint")
        if epoch is not None:
            raise ValueError("ZeroshotCLIP does not accept a checkpoint epoch")
        print("ZeroshotCLIP uses the frozen CLIP checkpoint directly")

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


# @TRAINER_REGISTRY.register()
# class ZeroshotCLIP2(ZeroshotCLIP):
#     """Prompt ensembling."""

#     # templates = IMAGENET_TEMPLATES
#     templates = IMAGENET_TEMPLATES_SELECT

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)
#         clip_model.to(self.device)

#         for params in clip_model.parameters():
#             params.requires_grad_(False)

#         # add custom-made prompt
#         if cfg.DATASET.NAME != "ImageNet":
#             self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

#         num_temp = len(self.templates)
#         print(f"Prompt ensembling (n={num_temp})")

#         mean_text_features = 0
#         for i, temp in enumerate(self.templates):
#             prompts = [temp.format(c.replace("_", " ")) for c in classnames]
#             prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
#             text_features = clip_model.encode_text(prompts)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             mean_text_features = mean_text_features + text_features
#         mean_text_features = mean_text_features / num_temp
#         mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

#         self.text_features = mean_text_features
#         self.clip_model = clip_model

