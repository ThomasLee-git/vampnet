from pathlib import Path
import datetime
import typing
import argparse


# torch
import torch
import torchaudio
from omegaconf import OmegaConf

from audiocraft.models.loaders import load_compression_model
from audiocraft.models.encodec import EncodecModel

import vampnet
from vampnet.modules.transformer import VampNet
from vampnet import mask as vampnet_mask
from vampnet import util as vampnet_util

IGNORE_INDEX = -100


class VampNetWrapper(torch.nn.Module):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        with open(config_path, mode="r") as rf:
            cfg = OmegaConf.create(rf.read())
        self.model = VampNet(**cfg.vampnet)

    def forward(self, inputs, sampled_dist):
        logits = self.model(inputs, sampled_dist)
        return logits

    def generate(self, **kwargs):
        return self.model.generate2(**kwargs)


def load_encodec(file_or_url: str = None, device="cpu"):
    if not file_or_url:
        file_or_url = "https://dl.fbaipublicfiles.com/audiocraft/musicgen/v0/b0dbef54-37d256b525.th"
    model = load_compression_model(file_or_url).to(device)
    return model


def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"{datetime.datetime.now()} done loading {model_path}")


def test_gen():
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    device = "cuda"
    encodec_model = load_encodec(device=device)
    encodec_weight_list = [
        encodec_model.quantizer.vq.layers[i]._codebook.embed
        for i in range(len(encodec_model.quantizer.vq.layers))
    ]
    # checkpoint_path = Path("last_checkpoints/unconditional_vampnet_model_30720.pth")
    # checkpoint_path = Path("last_checkpoints/unconditional_vampnet_model_104448.pth")
    checkpoint_path = Path("last_checkpoints/unconditional_vampnet_model_190464.pth")
    vampnet_config_path = Path("conf/vampnet_config.yaml")
    vampnet_wrapper = VampNetWrapper(vampnet_config_path).to(device)
    load_model(vampnet_wrapper, checkpoint_path, device)
    # construct tokens
    num_samples = 4
    num_sequences = 1500
    prompt_tokens = torch.zeros(
        (num_samples, encodec_model.num_codebooks, num_sequences), device=device
    ).long()
    mask = vampnet_mask.full_mask(prompt_tokens)
    prompt_tokens, mask = vampnet_mask.apply_mask(
        prompt_tokens, mask, mask_token=vampnet_wrapper.model.mask_token
    )
    generation_config = {
        "codec_weight_list": encodec_weight_list,
        "input_tokens": prompt_tokens,
        "mask": mask,
        "sampling_steps": 36,
        "temperature": 8.0,
    }
    generated_tokens = vampnet_wrapper.generate(**generation_config)
    # decode token
    wavs = encodec_model.decode(generated_tokens, None)
    for wav_idx, wav in enumerate(wavs):
        tmp_path = f"test_vampnet_gen_{wav_idx}.mp3"
        torchaudio.save(tmp_path, wav.cpu(), sample_rate=encodec_model.sample_rate)
    print("done")


if __name__ == "__main__":
    test_gen()
