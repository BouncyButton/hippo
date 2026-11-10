"""Small spatial encoder, independent prior fitting and frozen feature loss."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients
from scripts.depth_supervision.utils import atomic_json, digest, save_checkpoint, seed_all, rng_state, restore_rng


def one_hot(labels):
    return F.one_hot(labels.long(), 3).movedim(-1, 1).float()


class AnatomyEncoder(nn.Module):
    """Three spatial scales; no identity feature or unlearned distance term."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv3d(5, 8, 3, padding=1),
            nn.Conv3d(8, 12, 3, stride=2, padding=1), nn.Conv3d(12, 16, 3, stride=2, padding=1)])

    def forward(self, probabilities, context):
        # context = guide ROI, original FOV. Padding has no prediction gradient.
        fov = context[:, 1:2]
        background = torch.zeros_like(probabilities)
        background[:, :1] = 1
        p = probabilities * fov + background * (1 - fov)
        x = torch.cat([p, context], 1)
        result = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), .1)
            result.append(x)
        return result


class PriorTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AnatomyEncoder()
        self.up2 = nn.Conv3d(16 + 12, 12, 3, padding=1)
        self.up1 = nn.Conv3d(12 + 8, 8, 3, padding=1)
        self.candidate = nn.Conv3d(8, 3, 1)
        self.clean = nn.Conv3d(8, 3, 1)
        self.errors = nn.Conv3d(8, 4, 1)

    def forward(self, probabilities, context):
        fine, mid, coarse = self.encoder(probabilities, context)
        up = F.leaky_relu(self.up2(torch.cat([mid, F.interpolate(coarse, size=mid.shape[2:], mode='trilinear', align_corners=False)], 1)), .1)
        up = F.leaky_relu(self.up1(torch.cat([fine, F.interpolate(up, size=fine.shape[2:], mode='trilinear', align_corners=False)], 1)), .1)
        return self.candidate(up), self.clean(up), self.errors(up)


def balanced_ce(logits, target, fov):
    """Class balancing applies only to prior pretraining, not segmentation loss."""
    loss = F.cross_entropy(logits, target, reduction='none')
    mask = fov[:, 0].bool()
    return torch.stack([loss[(target == c) & mask].mean() for c in range(logits.shape[1])
                        if ((target == c) & mask).any()]).mean()


def field_of_view(shape, native_shape):
    out = np.zeros(shape, np.float32)
    start = (np.asarray(shape) - native_shape) // 2
    out[tuple(slice(int(a), int(a + n)) for a, n in zip(start, native_shape))] = 1
    return out


class ContextCache:
    def __init__(self, root, protocol, device):
        self.root, self.protocol, self.device = Path(root), protocol, device
        self.manifest = json.loads((self.root / 'roi_manifest.json').read_text())
        self.items = {}

    def get(self, item):
        name = item['case_name']
        if name not in self.items:
            path = self.root / 'roi' / (name + '.npz')
            assert digest(path) == self.manifest['files'][path.name]
            with np.load(path, allow_pickle=False) as a:
                roi = a['roi'].astype(np.float32)
            shape = tuple(item['label'].shape[1:])
            assert roi.shape == shape
            fov = field_of_view(shape, np.asarray(self.protocol['native_geometry'][name]['native_shape']))
            assert not np.any(roi * (1 - fov))
            self.items[name] = torch.from_numpy(np.stack([roi, fov]))[None].to(self.device)
        return self.items[name]


def load_training(root, protocol, device):
    names = protocol['splits']['5pct']['train']
    assert len(names) == 10 and not set(names) & set(protocol['splits']['5pct']['val'])
    # Only these rows are instantiated as Patients or used by prior fitting/audit.
    dataframe = _load_pkl_dataframe(protocol['pkl'])
    patients = Patients(dataframe, names, protocol['native_geometry'])
    cache = ContextCache(root, protocol, device)
    return patients, cache


class Corruptions:
    def __init__(self, item, context):
        self.truth = item['label'][0].numpy().astype(np.int64)
        self.fov = context[0, 1].cpu().numpy().astype(bool)
        self.affine = np.asarray(item['affine'])[:3, :3]
        self.centers = {}
        self.coordinates = np.indices(self.truth.shape).transpose(1, 2, 3, 0)
        for c in (1, 2):
            tissue = self.truth == c
            edge = tissue & ~ndimage.binary_erosion(tissue)
            spacing = np.linalg.norm(self.affine, axis=0)
            distance = ndimage.distance_transform_edt(~tissue, sampling=spacing)
            self.centers[c] = {
                'missing': np.argwhere(edge), 'swap': np.argwhere(tissue),
                'extra': np.argwhere((self.truth == 0) & self.fov & (distance <= 4.) & (distance > 0))}
            assert all(len(x) for x in self.centers[c].values())

    def make(self, kind, tissue_class, radius, rng):
        candidates = self.centers[tissue_class][kind]
        center = candidates[int(rng.integers(len(candidates)))]
        delta = (self.coordinates - center) @ self.affine.T
        sphere = (delta**2).sum(-1) <= radius**2
        selected = sphere & self.fov & (self.truth == (0 if kind == 'extra' else tissue_class))
        assert selected.any()
        wrong = self.truth.copy()
        wrong[selected] = {'missing': 0, 'extra': tissue_class, 'swap': 3 - tissue_class}[kind]
        return wrong, selected


def fit_prior(root, protocol, device):
    root = Path(root)
    out = root / 'encoder'
    out.mkdir(exist_ok=True)
    ident = dict(protocol=protocol['identity_sha'], train=protocol['splits']['5pct']['train'], settings=protocol['encoder'])
    complete = out / 'completion.json'
    if complete.exists():
        c = json.loads(complete.read_text())
        assert c['identity'] == ident and digest(out / 'frozen.pt') == c['encoder_sha256']
        return c
    seed_all(protocol['encoder']['seed'])
    patients, contexts = load_training(root, protocol, device)
    perturb = [Corruptions(item, contexts.get(item)) for item in patients.items]
    model = PriorTrainingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['encoder']['lr'], weight_decay=1e-5)
    rng = np.random.default_rng(protocol['encoder']['seed'])
    history, start = [], 0
    latest = out / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        assert state['identity'] == ident
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start, history = state['step'], state['history']
        rng.bit_generator.state = state['numpy_rng']; restore_rng(state['rng'])
    for step in range(start, protocol['encoder']['steps']):
        i = int(rng.integers(len(patients)))
        item, corrupt = patients[i], perturb[i]
        kind = ('missing', 'extra', 'swap')[step % 3]
        cls = int(rng.integers(1, 3))
        wrong, selected = corrupt.make(kind, cls, float(rng.choice([1.5, 3., 4.5])), rng)
        if step % 4 == 0:
            wrong, selected = corrupt.truth.copy(), np.zeros_like(selected)
        truth = item['label'].to(device).long()
        candidate = torch.from_numpy(wrong)[None].to(device)
        confidence = float(rng.uniform(.6, .99))
        probs = one_hot(candidate) * confidence + (1 - one_hot(candidate)) * ((1 - confidence) / 2)
        errors = torch.zeros_like(candidate)
        errors[torch.from_numpy(selected)[None].to(device)] = ('missing', 'extra', 'swap').index(kind) + 1
        context = contexts.get(item)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, clean, error_logits = model(probs, context)
        values = [balanced_ce(reconstruction, candidate, context[:, 1:2]),
                  balanced_ce(clean, truth, context[:, 1:2]), balanced_ce(error_logits, errors, context[:, 1:2])]
        loss = sum(values)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        history.append(dict(step=step+1, total=float(loss.detach()), candidate=float(values[0].detach()),
                            clean=float(values[1].detach()), errors=float(values[2].detach()), case_name=item['case_name']))
        if (step + 1) % 50 == 0 or step + 1 == protocol['encoder']['steps']:
            save_checkpoint(latest, dict(identity=ident, model=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step+1, history=history, numpy_rng=rng.bit_generator.state, rng=rng_state()))
            atomic_json(out / 'history.json', history)
            print(json.dumps(dict(stage='encoder', step=step+1, total=np.mean([r['total'] for r in history[-50:]]))), flush=True)
    model.eval()
    # Calibrate feature scales on clean training labels, never on validation data.
    rms = []
    with torch.no_grad():
        for item in patients.items:
            context = contexts.get(item)
            features = model.encoder(one_hot(item['label'].to(device)), context)
            rms.append([float((f.square() * F.interpolate(context[:, 1:2], size=f.shape[2:], mode='nearest')).sum() /
                (F.interpolate(context[:, 1:2], size=f.shape[2:], mode='nearest').sum() * f.shape[1])) for f in features])
    variances = np.mean(rms, axis=0).clip(1e-6).tolist()
    save_checkpoint(out / 'frozen.pt', dict(identity=ident, encoder=model.encoder.state_dict(), feature_variances=variances))
    c = dict(identity=ident, encoder_sha256=digest(out/'frozen.pt'), steps=len(history), train_cases=ident['train'],
        validation_cases_used=[], feature_variances=variances, encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        pretraining_parameters=sum(p.numel() for p in model.parameters()))
    atomic_json(complete, c)
    return c


class LearnedLoss(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.encoder = AnatomyEncoder().to(device)
        self.encoder.load_state_dict(state['encoder'])
        self.encoder.requires_grad_(False).eval()
        self.register_buffer('variances', torch.tensor(state['feature_variances'], device=device))
        self.target_cache = {}

    def features_loss(self, probabilities, labels, context, key=None):
        if key is not None and key in self.target_cache:
            targets = self.target_cache[key]
        else:
            with torch.no_grad():
                targets = self.encoder(one_hot(labels[:, 0]), context)
            if key is not None:
                self.target_cache[key] = targets
        features = self.encoder(probabilities.float(), context)
        terms = []
        for i, (a, b) in enumerate(zip(features, targets)):
            valid = F.interpolate(context[:, 1:2], size=a.shape[2:], mode='nearest')
            terms.append(((a-b).square() * valid).sum() / (valid.sum() * a.shape[1] * self.variances[i]))
        return torch.stack(terms).mean()

    def forward(self, logits, labels, context, key=None):
        return self.features_loss(logits.float().softmax(1), labels, context, key)
