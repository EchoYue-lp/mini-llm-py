import sys
from types import ModuleType

import pytest
import torch

from models.transformer_models import DecoderOnlyModel
from scripts.resume_training import (
    resume_decoder_training,
    resume_encoder_decoder_training,
)
from utils.checkpoint_utils import (
    get_checkpoint_config,
    get_checkpoint_training_config,
    load_checkpoint_for_training,
)
from utils.scheduler_utils import WarmupLRScheduler


def _decoder_config():
    return {
        'vocab_size': 32,
        'd_model': 16,
        'num_layers': 1,
        'num_heads': 4,
        'd_ff': 32,
        'max_len': 8,
        'dropout': 0.0,
    }


def test_resume_restores_model_optimizer_scheduler_and_epoch(tmp_path):
    torch.manual_seed(7)
    config = _decoder_config()
    model = DecoderOnlyModel(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupLRScheduler(
        optimizer,
        scheduler_type='cosine',
        num_training_steps=10,
        num_warmup_steps=2,
    )

    input_ids = torch.randint(0, config['vocab_size'], (2, 6))
    loss = model(input_ids)[0].square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint_path = tmp_path / 'decoder.pt'
    training_config = {
        'use_scheduler': True,
        'scheduler_type': 'cosine',
        'num_training_steps': 10,
        'num_warmup_steps': 2,
    }
    torch.save(
        {
            'epoch': 3,
            'val_loss': 1.25,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'training_config': training_config,
        },
        checkpoint_path,
    )

    restored_model = DecoderOnlyModel(**config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = WarmupLRScheduler(
        restored_optimizer,
        scheduler_type='cosine',
        num_training_steps=10,
        num_warmup_steps=2,
    )
    info = load_checkpoint_for_training(
        str(checkpoint_path),
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    assert info == {'start_epoch': 4, 'best_val_loss': 1.25}
    assert get_checkpoint_config(str(checkpoint_path), require_saved=True) == config
    assert get_checkpoint_training_config(str(checkpoint_path)) == training_config
    assert restored_optimizer.state_dict()['state']
    assert restored_scheduler.state_dict()['last_epoch'] == scheduler.state_dict()['last_epoch']
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_resume_rejects_weight_only_checkpoint_without_config(tmp_path):
    checkpoint_path = tmp_path / 'weights_only.pt'
    torch.save(DecoderOnlyModel(**_decoder_config()).state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match='缺少 config'):
        get_checkpoint_config(str(checkpoint_path), require_saved=True)


def test_decoder_training_resumes_with_checkpoint_architecture(
    tmp_path,
    monkeypatch,
):
    from scripts import train_decoder

    class FakeTokenizer:
        pad_token_id = 31

        def __len__(self):
            return 32

    class DummyWriter:
        def __init__(self, **kwargs):
            self.log_dir = kwargs['log_dir']

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def close(self):
            pass

    tensorboard_module = ModuleType('torch.utils.tensorboard')
    tensorboard_module.SummaryWriter = DummyWriter
    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', tensorboard_module)
    monkeypatch.setattr(
        train_decoder,
        'load_gpt2_tokenizer',
        lambda tokenizer_dir: FakeTokenizer(),
    )
    monkeypatch.setattr(
        train_decoder,
        'load_dataset_pt',
        lambda path: [[1, 2, 3, 4], [5, 6, 7]],
    )

    config = _decoder_config()
    model = DecoderOnlyModel(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / 'resume.pt'
    torch.save(
        {
            'epoch': 1,
            'val_loss': float('inf'),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'training_config': {'use_scheduler': False},
        },
        checkpoint_path,
    )

    monkeypatch.chdir(tmp_path)
    train_decoder.train_decoder_only(
        epochs=1,
        batch_size=2,
        device='cpu',
        resume_from=str(checkpoint_path),
        log_root=str(tmp_path / 'runs'),
    )

    resumed = torch.load(
        tmp_path / 'decoder_only_best.pt',
        map_location='cpu',
        weights_only=False,
    )
    assert resumed['epoch'] == 2
    assert resumed['config'] == config


def test_resume_helpers_call_real_training_entrypoints(monkeypatch):
    calls = {}

    def fake_decoder(**kwargs):
        calls['decoder'] = kwargs
        return 'decoder-result'

    def fake_encoder_decoder(**kwargs):
        calls['encoder_decoder'] = kwargs
        return 'encoder-decoder-result'

    decoder_module = ModuleType('scripts.train_decoder')
    decoder_module.train_decoder_only = fake_decoder
    encoder_decoder_module = ModuleType('scripts.train_encoder_decoder')
    encoder_decoder_module.train_encoder_decoder = fake_encoder_decoder
    monkeypatch.setitem(sys.modules, 'scripts.train_decoder', decoder_module)
    monkeypatch.setitem(
        sys.modules,
        'scripts.train_encoder_decoder',
        encoder_decoder_module,
    )

    assert resume_decoder_training('decoder.pt', additional_epochs=4) == 'decoder-result'
    assert (
        resume_encoder_decoder_training('encoder.pt', additional_epochs=5)
        == 'encoder-decoder-result'
    )
    assert calls['decoder']['resume_from'] == 'decoder.pt'
    assert calls['decoder']['epochs'] == 4
    assert calls['encoder_decoder']['resume_from'] == 'encoder.pt'
    assert calls['encoder_decoder']['epochs'] == 5
