"""Geometry, RNG isolation, and interrupted-training coverage for mild_v1."""

from contextlib import ExitStack
import json
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from thesis.new_constraints.training_augmentation import augment_mild, warp_pair
from thesis.new_constraints import train_swinunetr_constraints as trainer
from thesis.new_constraints.test_followup_training import _TinySegmentationModel, _records


def pair(size=16):
    labels = torch.zeros(1, 1, size, size, size, dtype=torch.long)
    labels[:, :, 5:8, 5:10, 5:10] = 1
    labels[:, :, 8:11, 5:10, 5:10] = 2
    return labels.float(), labels


def test_affine_integer_translation_keeps_image_label_alignment():
    images, labels = pair()
    image, label, applied = warp_pair(images, labels, angles=(0, 0, 0), scale=1,
                                     translation_xyz=(1, -1, 0))
    assert applied
    expected = labels.roll(1, -1).roll(-1, -2)
    assert torch.equal(label, expected)
    torch.testing.assert_close(image, expected.float(), atol=2e-6, rtol=0)


def test_affine_rejects_foreground_clipping_without_mutation():
    images, labels = pair()
    image, label, applied = warp_pair(images, labels, angles=(0, 0, 0), scale=1,
                                     translation_xyz=(20, 0, 0))
    assert not applied and image is images and label is labels


def test_augmentation_rng_replay_global_isolation_and_valid_labels():
    images, labels = pair()
    original_image, original_label = images.clone(), labels.clone()
    generator = torch.Generator().manual_seed(73)
    initial = generator.get_state()
    global_state = torch.get_rng_state()
    outputs = [augment_mild(images, labels, generator=generator) for _ in range(30)]
    assert torch.equal(torch.get_rng_state(), global_state)
    generator.set_state(initial)
    replay = [augment_mild(images, labels, generator=generator) for _ in range(30)]
    for (image, label, stats), (other_image, other_label, other_stats) in zip(outputs, replay):
        assert torch.equal(image, other_image) and torch.equal(label, other_label)
        assert stats == other_stats and torch.isfinite(image).all()
        assert label.dtype == labels.dtype and set(label.unique().tolist()) == {0, 1, 2}
    assert any(stats['identity'] for _, _, stats in outputs)
    assert any(stats['spatial_applied'] for _, _, stats in outputs)
    assert any(stats['noise'] for _, _, stats in outputs)
    assert any(stats['blur'] for _, _, stats in outputs)
    assert torch.equal(images, original_image) and torch.equal(labels, original_label)


@pytest.mark.parametrize('extra', [
    ['--batch-size', '2'], ['--translation-augmentation'], ['--constraint-set', 'equivariance'],
])
def test_augmentation_rejects_unsupported_combinations(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, 'argv', ['train', '--pkl', 'unused', '--splits-json', 'unused',
        '--output-dir', str(tmp_path), '--constraint-set', 'none', '--training-augmentation', 'mild_v1', *extra])
    with pytest.raises(ValueError, match='mild_v1 requires'):
        with ExitStack() as stack:
            trainer._main(stack)


def test_augmented_training_resume_matches_uninterrupted_and_validation_is_clean(monkeypatch, tmp_path):
    pkl, splits = tmp_path / 'data.pkl', tmp_path / 'splits.json'
    pkl.write_bytes(b'fixture')
    splits.write_text(json.dumps([{'train': ['train_0', 'train_1'], 'val': ['val_0', 'val_1']}]))
    monkeypatch.setattr(trainer, 'build_swinunetr', lambda *args, **kwargs: _TinySegmentationModel().to(args[2]))
    def build_data(args, generator, **kwargs):
        return (DataLoader(_records('train'), batch_size=1, shuffle=True, generator=generator),
                DataLoader(_records('val'), batch_size=1), 3, 2, 2)
    monkeypatch.setattr(trainer, 'build_data', build_data)
    monkeypatch.setattr(trainer, 'collect_source_provenance', lambda: {'sha256': 'f'*64, 'files': {}})
    monkeypatch.setattr(trainer, 'validate_source_provenance_unchanged', lambda *args: None)
    calls = []
    real_augment = trainer.augment_mild
    def counted(*args, **kwargs):
        calls.append(1)
        return real_augment(*args, **kwargs)
    monkeypatch.setattr(trainer, 'augment_mild', counted)
    def run(path, resume=False, augmentation='mild_v1'):
        monkeypatch.setattr(sys, 'argv', ['train', '--pkl', str(pkl), '--splits-json', str(splits),
            '--output-dir', str(path), '--constraint-set', 'none', '--training-augmentation', augmentation,
            '--epochs', '3', '--spatial-size', '8', '8', '8', '--device', 'cpu', '--plain-tensors',
            *(['--resume'] if resume else [])])
        with ExitStack() as stack:
            trainer._main(stack)
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    run(full)
    assert len(calls) == 6  # two training cases x three epochs; never validation/final evaluation
    real_save = trainer.save_checkpoint
    def interrupt(path, **kwargs):
        real_save(path, **kwargs)
        if path.name == 'checkpoint_latest.pt' and kwargs['epoch'] == 1:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(trainer, 'save_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(interrupted)
    monkeypatch.setattr(trainer, 'save_checkpoint', real_save)
    run(interrupted, resume=True)
    assert len(calls) == 12
    a = torch.load(full / 'checkpoint_latest.pt', weights_only=True)
    b = torch.load(interrupted / 'checkpoint_latest.pt', weights_only=True)
    assert all(torch.equal(value, b['model'][key]) for key, value in a['model'].items())
    assert (full / 'metrics.csv').read_text() == (interrupted / 'metrics.csv').read_text()
    assert a['run']['training_augmentation'] == 'mild_v1'
    with pytest.raises(ValueError, match='Resume configuration does not match'):
        run(interrupted, resume=True, augmentation='none')
