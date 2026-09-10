"""Shared, device-neutral TextConsistency primitives for controlled experiments.

Groups are contiguous blocks of descriptions, not annotated semantic categories.
"""
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_FILES = {
    'Caltech101': 'templates/cal_prompts_full.json',
    'OxfordPets': 'templates/pets_prompts_full.json',
    'StanfordCars': 'templates/cars_prompts_full.json',
    'OxfordFlowers': 'templates/flower_prompts_full.json',
    'Food101': 'templates/food_prompts_full.json',
    'FGVCAircraft': 'templates/airplane_prompts_full.json',
    'SUN397': 'templates/sun_prompts_full.json',
    'DescribableTextures': 'templates/texture_prompts_full.json',
    'EuroSAT': 'templates/eurosat_prompts_full.json',
    'UCF101': 'templates/ucf_prompts_full.json',
    **{key: 'templates/imagenet_prompts_full.json' for key in
       ('ImageNet', 'ImageNetV2', 'ImageNetSketch', 'ImageNetA', 'ImageNetR')},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_device(cfg):
    name = cfg.TRAINER.TEXTCONSISTENCY.DEVICE
    if name == 'auto':
        name = 'cuda' if cfg.USE_CUDA and torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if name not in ('cpu', 'cuda'):
        raise ValueError('Supported reproducibility devices: auto, cpu, cuda')
    return torch.device(name)


def prediction_logits(student, teacher, branch, alpha=0.5):
    if branch not in ('student', 'teacher', 'ensemble'):
        raise ValueError(f'Unknown inference branch: {branch}')
    if not 0 <= alpha <= 1:
        raise ValueError('ENSEMBLE_ALPHA must be in [0, 1]')
    if teacher is None:
        if branch != 'student':
            raise ValueError('This baseline only supports student inference')
        return student
    if branch == 'student':
        return student
    if branch == 'teacher':
        return teacher
    return (1 - alpha) * student + alpha * teacher


def resolve_checkpoint(directory, name, epoch=None):
    if not directory:
        raise FileNotFoundError('Evaluation requires a checkpoint directory')
    folder = Path(directory) / name
    if epoch is not None:
        path = folder / f'model.pth.tar-{epoch}'
    elif (folder / 'model-best.pth.tar').is_file():
        path = folder / 'model-best.pth.tar'
    else:
        candidates = [p for p in folder.glob('model.pth.tar-*') if p.name.rsplit('-', 1)[-1].isdigit()]
        if not candidates:
            raise FileNotFoundError(f'No complete checkpoint in {folder}; old prompt-only weights are insufficient')
        path = max(candidates, key=lambda p: int(p.name.rsplit('-', 1)[-1]))
    if not path.is_file():
        raise FileNotFoundError(f'Requested checkpoint does not exist: {path}')
    return path


def load_complete_checkpoint(trainer, directory, epoch=None):
    records = []
    for name in trainer.get_model_names():
        path = resolve_checkpoint(directory, name, epoch)
        # These are local experiment checkpoints (including optimizer state).
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if epoch is not None and checkpoint.get('epoch') != epoch:
            raise ValueError('Checkpoint epoch does not match the requested epoch')
        saved_spec = checkpoint.get('repro_signature')
        expected_spec = getattr(trainer, 'repro_signature', None)
        if expected_spec is not None and saved_spec != expected_spec:
            raise ValueError('Checkpoint model/data-text signature mismatch; use the matching configuration')
        current_provenance = getattr(trainer, 'run_provenance', None)
        if current_provenance is not None:
            saved_provenance = checkpoint.get('run_provenance', {})
            for field in ('code_sha256', 'seed', 'shots', 'effective_losses'):
                if saved_provenance.get(field) != current_provenance[field]:
                    raise ValueError(f'Checkpoint training provenance mismatch: {field}')
            source_dataset, target_dataset = saved_provenance.get('dataset'), current_provenance['dataset']
            if source_dataset != target_dataset and not (source_dataset == 'ImageNet' and target_dataset.startswith('ImageNet')):
                raise ValueError('Checkpoint dataset mismatch')
            if current_provenance['subsample'] == 'base' and saved_provenance.get('train_samples_sha256') != current_provenance['train_samples_sha256']:
                raise ValueError('Checkpoint few-shot training split mismatch')
        model = trainer._models[name]
        state = {k: v for k, v in checkpoint['state_dict'].items()
                 if not k.endswith(('token_prefix', 'token_suffix'))}
        expected = {k: v for k, v in model.state_dict().items()
                    if not k.endswith(('token_prefix', 'token_suffix'))}
        missing, extra = set(expected) - set(state), set(state) - set(expected)
        if missing or extra:
            raise ValueError(f'Incomplete checkpoint; missing={sorted(missing)}, unexpected={sorted(extra)}')
        if any(expected[k].shape != state[k].shape for k in expected):
            raise ValueError('Checkpoint tensor shapes do not match the model')
        model.load_state_dict(state, strict=False)  # only class-dependent prompt buffers are rebuilt
        record = {'path': str(path.resolve()), 'sha256': sha256_file(path), 'epoch': checkpoint['epoch']}
        records.append(record)
        print('Loaded checkpoint: ' + json.dumps(record, sort_keys=True))
    trainer.loaded_checkpoints = records
    return records


def _masked_softmax(logits, mask, dim=-1):
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class ViewAggregator(nn.Module):
    def __init__(self, dim, n_views, mode='hierarchical', n_groups=4,
                 use_feature_adapter=True, hidden_dim=None):
        super().__init__()
        if mode not in ('mean', 'flat', 'flat_dual', 'hierarchical') or n_views < 1 or n_groups < 1:
            raise ValueError('Invalid aggregation configuration')
        if mode == 'mean' and use_feature_adapter:
            raise ValueError('A query adapter has no effect on mean aggregation')
        self.mode, self.n_groups, self.n_views = mode, n_groups, n_views
        hidden_dim = hidden_dim or max(1, dim // 4)
        self.fine_grained_proj = (nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                                  if use_feature_adapter else None)
        if mode != 'mean':
            self.intra_group_query = nn.Linear(dim, hidden_dim)
            self.intra_group_key = nn.Linear(dim, hidden_dim)
            self.intra_group_temp = nn.Parameter(torch.tensor([0.07]))
        if mode in ('hierarchical', 'flat_dual'):
            self.inter_group_query = nn.Linear(dim, hidden_dim)
            self.inter_group_key = nn.Linear(dim, hidden_dim)
            self.inter_group_temp = nn.Parameter(torch.tensor([0.07]))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image_features, multiview_features, view_mask=None):
        image, features = image_features.float(), multiview_features.float()
        n_cls, n_views, dim = features.shape
        mask = (torch.ones((n_cls, n_views), dtype=torch.bool, device=features.device)
                if view_mask is None else view_mask.bool())
        if mask.shape != features.shape[:2] or not mask.any(dim=-1).all():
            raise ValueError('Each class requires at least one valid description')
        if self.fine_grained_proj is not None:
            image = F.normalize(image + self.fine_grained_proj(image), dim=-1)
        if self.mode == 'mean':
            weights = mask.float() / mask.sum(-1, keepdim=True)
            result = torch.einsum('cv,cvd->cd', weights, features)
            return F.normalize(result, dim=-1).unsqueeze(0).expand(image.shape[0], -1, -1).to(image_features.dtype), {}
        query = self.intra_group_query(image)
        if self.mode in ('flat', 'flat_dual'):
            keys = self.intra_group_key(features)
            logits = torch.einsum('bh,cvh->bcv', query, keys)
            weights = _masked_softmax(logits / self.intra_group_temp.abs().clamp_min(0.01), mask[None])
            if self.mode == 'flat_dual':
                # Two global heads: exactly the hierarchical projection/temperature
                # budget, with no partitioning or unused capacity-matching weights.
                second_logits = torch.einsum('bh,cvh->bcv', self.inter_group_query(image),
                                              self.inter_group_key(features))
                second = _masked_softmax(second_logits / self.inter_group_temp.abs().clamp_min(0.01), mask[None])
                weights = 0.5 * (weights + second)
            result = torch.einsum('bcv,cvd->bcd', weights, features)
            return F.normalize(result, dim=-1).to(image_features.dtype), {'attn': weights}
        groups = min(self.n_groups, n_views)
        width = math.ceil(n_views / groups)
        pad = groups * width - n_views
        grouped = F.pad(features, (0, 0, 0, pad)).reshape(n_cls, groups, width, dim)
        grouped_mask = F.pad(mask, (0, pad), value=False).reshape(n_cls, groups, width)
        intra_logits = torch.einsum('bh,cgvh->bcgv', query, self.intra_group_key(grouped))
        intra = _masked_softmax(intra_logits / self.intra_group_temp.abs().clamp_min(0.01), grouped_mask[None])
        pooled = torch.einsum('bcgv,cgvd->bcgd', intra, grouped)
        inter_logits = torch.einsum('bh,bcgh->bcg', self.inter_group_query(image), self.inter_group_key(pooled))
        inter = _masked_softmax(inter_logits / self.inter_group_temp.abs().clamp_min(0.01), grouped_mask.any(-1)[None])
        result = torch.einsum('bcg,bcgd->bcd', inter, pooled)
        return F.normalize(result, dim=-1).to(image_features.dtype), {'intra_attn': intra, 'inter_attn': inter}


def build_view_aggregator(cfg, dim):
    tc = cfg.TRAINER.TEXTCONSISTENCY
    return ViewAggregator(dim, tc.N_VIEWS, tc.AGGREGATION, tc.N_GROUPS, tc.USE_FEATURE_ADAPTER)


def description_rows(templates, classnames, n_views):
    if n_views < 1:
        raise ValueError('N_VIEWS must be positive')
    def canonical(name):
        return ' '.join(name.replace('_', ' ').casefold().split())
    lookup = {}
    for key, value in templates.items():
        normalized = canonical(key)
        if normalized in lookup and lookup[normalized] != value:
            raise ValueError(f'Ambiguous normalized class name: {key}')
        lookup[normalized] = value
    rows = []
    for name in classnames:
        candidates = lookup.get(canonical(name))
        if not isinstance(candidates, list) or any(not isinstance(x, str) for x in candidates):
            raise ValueError(f'Missing or invalid descriptions for class: {name}')
        # Padding is applied only to tensors with an explicit mask, never by repeating text.
        unique = list(dict.fromkeys(' '.join(x.split()) for x in candidates if x.strip()))[:n_views]
        if not unique:
            raise ValueError(f'No nonempty descriptions for class: {name}')
        rows.append(unique)
    return rows


def description_permutation(count, policy='contiguous', n_groups=4, seed=0):
    """Reorder only effective views, with an RNG independent of training seeds.

    Interleaved order gathers positions 0,G,2G,... then 1,1+G,... before
    the unchanged contiguous hierarchical pooling. No semantic labels are used.
    """
    if count < 1 or n_groups < 1 or not isinstance(seed, int) or seed < 0:
        raise ValueError('Invalid description order configuration')
    if policy not in ('contiguous', 'interleaved', 'shuffle') or (policy != 'shuffle' and seed != 0):
        raise ValueError('Use contiguous/interleaved with seed 0, or shuffle with an explicit seed')
    order = list(range(count))
    if policy == 'interleaved':
        order.sort(key=lambda index: (index % n_groups, index // n_groups))
    elif policy == 'shuffle':
        random.Random(seed).shuffle(order)
    return order


def view_order_manifest(classnames, counts, policy, n_groups, seed):
    records = [{'classname': name, 'count': count,
                'order': description_permutation(count, policy, n_groups, seed)}
               for name, count in zip(classnames, counts)]
    if len(classnames) != len(counts) or len(set(classnames)) != len(classnames):
        raise ValueError('Description ordering requires unique classes and complete counts')
    payload = {'version': 1, 'policy': policy, 'seed': seed, 'n_groups': n_groups, 'classes': records}
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


class CachedMultiViewTextEncoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        from clip import clip
        tc = cfg.TRAINER.TEXTCONSISTENCY
        relative = TEMPLATE_FILES.get(cfg.DATASET.NAME)
        if relative is None:
            raise ValueError(f'No template mapping for {cfg.DATASET.NAME}')
        path = PROJECT_ROOT / relative
        rows = description_rows(json.loads(path.read_text()), classnames, tc.N_VIEWS)
        self.template_sha256 = sha256_file(path)
        self.description_counts = [len(row) for row in rows]
        self.classnames = list(classnames)
        self.n_views = tc.N_VIEWS
        device = select_device(cfg)
        original_device = next(clip_model.parameters()).device
        clip_model.to(device).eval()
        flat = [text for row in rows for text in row]
        tokens = clip.tokenize(flat, truncate=True)
        # Token truncation can make descriptions identical; deduplicate token rows
        # and record the resulting effective counts.
        encoded = []
        with torch.no_grad():
            for chunk in tokens.split(tc.TEXT_BATCH_SIZE):
                features = clip_model.encode_text(chunk.to(device)).float()
                encoded.append(F.normalize(features, dim=-1).cpu())
        all_features = torch.cat(encoded)
        features = torch.zeros(len(rows), tc.N_VIEWS, all_features.shape[-1])
        mask = torch.zeros(len(rows), tc.N_VIEWS, dtype=torch.bool)
        offset = 0
        token_counts = []
        for i, row in enumerate(rows):
            count = len(row)
            seen, indices = set(), []
            for j in range(count):
                key = tuple(tokens[offset+j].tolist())
                if key not in seen:
                    seen.add(key); indices.append(j)
            token_counts.append(len(indices))
            features[i, :len(indices)] = all_features[offset:offset+count][indices]
            mask[i, :len(indices)] = True
            offset += count
        self.description_counts = token_counts
        self.view_order_manifest = view_order_manifest(
            self.classnames, token_counts, tc.VIEW_ORDER, tc.N_GROUPS, tc.ORDER_SEED)
        # Encode/deduplicate in original order first. Reorder valid entries only,
        # preserving description membership and the exact padding/group occupancy.
        if tc.VIEW_ORDER != 'contiguous':
            for i, row in enumerate(self.view_order_manifest['classes']):
                features[i, :row['count']] = features[i, row['order']].clone()
        self.register_buffer('multiview_features', features)
        self.register_buffer('view_mask', mask)
        clip_model.to(original_device)
        print('Effective description counts: ' + json.dumps(dict(zip(classnames, token_counts)), sort_keys=True))

    def get_view_features(self):
        return self.multiview_features

    def get_single_view_features(self):
        return self.multiview_features[:, 0]


def training_losses(student, teacher, labels, attention, kd_weight, kd_temp,
                    adaptive_ce_weight, diversity_weight, contrastive_weight=0.):
    if contrastive_weight != 0:
        raise ValueError('CONTRASTIVE_WEIGHT must be 0: the previous frozen-feature loss is disabled')
    if kd_temp <= 0 or min(kd_weight, adaptive_ce_weight, diversity_weight) < 0:
        raise ValueError('Invalid loss weights or KD temperature')
    ce = F.cross_entropy(student, labels)
    zero = ce.new_zeros(())
    adaptive_ce = F.cross_entropy(teacher, labels) if teacher is not None else zero
    kd = (F.kl_div(F.log_softmax(student / kd_temp, dim=-1),
                   F.softmax(teacher.detach() / kd_temp, dim=-1), reduction='batchmean') * kd_temp**2
          if teacher is not None and kd_weight else zero)
    neg_entropy = zero
    for name in ('attn', 'intra_attn', 'inter_attn'):
        if attention is not None and name in attention:
            weights = attention[name]
            valid = weights.sum(-1) > 0
            terms = (weights * weights.clamp_min(1e-12).log()).sum(-1)
            neg_entropy = neg_entropy + terms[valid].mean()
    total = ce + adaptive_ce_weight * adaptive_ce + kd_weight * kd + diversity_weight * neg_entropy
    return total, {'loss_ce': ce, 'loss_adaptive_ce': adaptive_ce, 'loss_kd': kd, 'loss_negative_entropy': neg_entropy}


def load_clip_checkpoint(cfg):
    from clip import clip
    tc = cfg.TRAINER.TEXTCONSISTENCY
    path = Path(tc.CLIP_PATH).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if cfg.MODEL.BACKBONE.NAME != 'ViT-B/16':
        raise ValueError('This reproduction configuration is validated only for ViT-B/16')
    if not path.is_file():
        raise FileNotFoundError(f'CLIP checkpoint missing: {path}; run scripts/repro/prepare_assets.py')
    digest = sha256_file(path)
    expected = clip._MODELS['ViT-B/16'].split('/')[-2]
    if digest != expected:
        raise ValueError('CLIP checkpoint SHA-256 does not match the official ViT-B/16 weights')
    jit_model = torch.jit.load(str(path), map_location='cpu').eval()
    details = {'trainer': 'CoOp', 'vision_depth': 0, 'language_depth': 0, 'vision_ctx': 0, 'language_ctx': 0}
    model = clip.build_model(jit_model.state_dict(), details).float().eval()
    model._repro_sha256 = digest
    return model
