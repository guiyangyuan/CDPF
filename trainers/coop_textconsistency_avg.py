"""
TextConsistency Avg: Multi-View with Simple Averaging (Ablation Study)

用于消融实验：使用多视角文本，但简单平均（不用自适应加权）
"""

import os.path as osp
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

TEMPLATE_FILES = {
    "Caltech101": "templates/cal_prompts_full.json",
    "OxfordPets": "templates/pets_prompts_full.json",
    "StanfordCars": "templates/cars_prompts_full.json",
    "OxfordFlowers": "templates/flower_prompts_full.json",
    "Food101": "templates/food_prompts_full.json",
    "FGVCAircraft": "templates/airplane_prompts_full.json",
    "SUN397": "templates/sun_prompts_full.json",
    "DescribableTextures": "templates/texture_prompts_full.json",
    "EuroSAT": "templates/eurosat_prompts_full.json",
    "UCF101": "templates/ucf_prompts_full.json",
    "ImageNet": "templates/imagenet_prompts_full.json",
}


def load_clip_to_cpu(cfg):
    model_path = './clip/ViT-B-16.pt'
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp', "vision_depth": 0, "language_depth": 0,
                      "vision_ctx": 0, "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.TEXTCONSISTENCY.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts


class MultiViewTextEncoder(nn.Module):
    """多视角文本编码器"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_views = cfg.TRAINER.TEXTCONSISTENCY.N_VIEWS
        self._load_multiview_features(cfg, classnames, clip_model)
    
    def _load_multiview_features(self, cfg, classnames, clip_model):
        dataset_name = cfg.DATASET.NAME
        template_file = TEMPLATE_FILES.get(dataset_name)
        
        if template_file is None or not osp.exists(template_file):
            self._create_default_features(classnames, clip_model)
            return
        
        with open(template_file, 'r') as f:
            templates = json.load(f)
        
        clip_model_cuda = load_clip_to_cpu(cfg).cuda()
        clip_model_cuda.eval()
        
        all_view_features = []
        
        with torch.no_grad():
            for classname in classnames:
                key = classname.replace("_", " ").lower()
                descriptions = None
                
                for k in [key, classname, classname.replace("_", " ")]:
                    if k in templates:
                        descriptions = templates[k][:self.n_views]
                        break
                
                if descriptions is None or len(descriptions) == 0:
                    descriptions = [f"a photo of a {classname.replace('_', ' ')}"]
                
                while len(descriptions) < self.n_views:
                    descriptions.append(descriptions[0])
                descriptions = descriptions[:self.n_views]
                
                tokens = clip.tokenize(descriptions, truncate=True).cuda()
                features = clip_model_cuda.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                all_view_features.append(features)
        
        multiview_features = torch.stack(all_view_features, dim=0)
        self.register_buffer("multiview_features", multiview_features)
        
        # 简单平均所有视角
        mean_features = multiview_features.mean(dim=1)
        mean_features = mean_features / mean_features.norm(dim=-1, keepdim=True)
        self.register_buffer("mean_features", mean_features)
        
        print(f"[Ablation-Avg] Loaded {self.n_views} views for {self.n_cls} classes (Simple Averaging)")
        
        del clip_model_cuda
        torch.cuda.empty_cache()
    
    def _create_default_features(self, classnames, clip_model):
        clip_model_cuda = load_clip_to_cpu(None).cuda()
        clip_model_cuda.eval()
        
        with torch.no_grad():
            texts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
            tokens = clip.tokenize(texts).cuda()
            features = clip_model_cuda.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        
        multiview_features = features.unsqueeze(1).expand(-1, self.n_views, -1).clone()
        self.register_buffer("multiview_features", multiview_features)
        self.register_buffer("mean_features", features)
        
        del clip_model_cuda
        torch.cuda.empty_cache()
    
    def get_mean_features(self):
        """返回简单平均的特征"""
        return self.mean_features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.multiview_encoder = MultiViewTextEncoder(cfg, classnames, clip_model)
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        # 简单平均的多视角特征
        mean_features = self.multiview_encoder.get_mean_features()
        logits_avg = logit_scale * image_features @ mean_features.t()
        
        return logits, logits_avg


@TRAINER_REGISTRY.register()
class CoOp_TextConsistency_Avg(TrainerX):
    """
    Ablation Study: Multi-View with Simple Averaging
    用于证明多视角的价值（vs 单视角baseline）
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TEXTCONSISTENCY.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.TEXTCONSISTENCY.PREC == "fp32" or cfg.TRAINER.TEXTCONSISTENCY.PREC == "amp":
            clip_model.float()

        print("Building TextConsistency Avg CLIP (Ablation: Simple Averaging)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in image and text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)

        self.optim = torch.optim.SGD(
            self.model.prompt_learner.parameters(),
            lr=cfg.OPTIM.LR,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.TEXTCONSISTENCY.PREC == "amp" else None
        
        self.kd_weight = cfg.TRAINER.TEXTCONSISTENCY.KD_WEIGHT
        self.kd_temp = cfg.TRAINER.TEXTCONSISTENCY.KD_TEMP

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        logits, logits_avg = self.model(image)
        
        loss_ce = F.cross_entropy(logits, label)
        
        if self.kd_weight > 0:
            loss_kd = F.kl_div(
                F.log_softmax(logits / self.kd_temp, dim=1),
                F.softmax(logits_avg.detach() / self.kd_temp, dim=1),
                reduction='batchmean'
            ) * (self.kd_temp ** 2)
        else:
            loss_kd = torch.tensor(0.0).to(self.device)
        
        loss = loss_ce + self.kd_weight * loss_kd
        
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_kd": loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd,
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def model_inference(self, image):
        logits, _ = self.model(image)
        return logits

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                import glob
                pattern = osp.join(directory, name, "model.pth.tar-*")
                model_files = glob.glob(pattern)
                if model_files:
                    model_files.sort(key=lambda x: int(x.split("-")[-1]))
                    model_path = model_files[-1]
                    print(f"model-best not found, using {model_path}")
                else:
                    print(f"No model found for {name}, skipping")
                    continue

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
