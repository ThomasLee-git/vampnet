from pathlib import Path
import math


import numpy as np
import torch
import torchaudio

from lac.model.lac import LAC as DAC
from vampnet.modules.transformer import VampNet, CodebookEmbedding
import vampnet.mask as vampnet_mask


def load_codec(path: str, device="cpu"):
    codec = DAC.load(Path(path))
    codec.to(device)
    codec.eval()
    return codec


def _load_model(
    ckpt: str,
    lora_ckpt: str = None,
    device: str = "cpu",
    chunk_size_s: int = 10,
):
    # we need to set strict to False if the model has lora weights to add later
    model = VampNet.load(location=Path(ckpt), map_location="cpu", strict=False)

    # load lora weights if needed
    if lora_ckpt is not None:
        if not Path(lora_ckpt).exists():
            should_cont = input(
                f"lora checkpoint {lora_ckpt} does not exist. continue? (y/n) "
            )
            if should_cont != "y":
                raise Exception("aborting")
        else:
            model.load_state_dict(
                torch.load(lora_ckpt, map_location="cpu"), strict=False
            )

    model.to(device)
    model.eval()
    model.chunk_size_s = chunk_size_s
    return model


def token2latents(
    tokens: torch.Tensor, embedding_model: CodebookEmbedding, codec_model: DAC
):
    with torch.no_grad():
        latents = embedding_model.from_codes(tokens, codec_model)
    return latents


def latents2embeddings(latents: torch.Tensor, codec_model: DAC):
    with torch.no_grad():
        ret = codec_model.quantizer.from_latents(latents)
    return ret[0]


def token2embeddings(tokens, codec_model: DAC):
    with torch.no_grad():
        ret = codec_model.quantizer.from_codes(tokens)
    return ret[0]


def generate(output_root):
    codec_path = "vampnet_checkpoints/codec.pth"
    coarse_path = "vampnet_checkpoints/coarse.pth"
    c2f_path = "vampnet_checkpoints/c2f.pth"
    device = "cuda:0"
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    codec_model = load_codec(codec_path, device=device)
    coarse_model = _load_model(coarse_path, device=device, chunk_size_s=10)
    c2f_model = _load_model(c2f_path, device=device, chunk_size_s=3)

    # generate tokens and mask
    num_samples = 1
    audio_seconds = 30.0
    sample_len = int(audio_seconds * codec_model.sample_rate)
    dummy_audio = torch.zeros((num_samples, 1, sample_len), device=device)
    encoded_dict = codec_model.encode(dummy_audio)
    tokens = encoded_dict["codes"]
    mask = torch.ones_like(tokens, device=device)
    tokens, mask = vampnet_mask.apply_mask(tokens, mask, coarse_model.mask_token)

    # coarse stage
    coarse_tokens = tokens[:, : coarse_model.n_codebooks, :]
    coarse_mask = mask[:, coarse_model.n_codebooks, :]
    coarse_tokens = coarse_model.generate(
        codec_model,
        start_tokens=coarse_tokens,
        mask=coarse_mask,
        sampling_steps=36,
        temperature=8.0,
        return_signal=False,
    )

    coarse_tokens_path = output_root.joinpath(f"vampnet_{audio_seconds}_coarse_tokens")
    np.save(coarse_tokens_path, coarse_tokens.cpu().numpy())

    # assign
    tokens[:, : coarse_model.n_codebooks, :] = coarse_tokens
    mask[:, : coarse_model.n_codebooks, :] = 0

    # fine stage
    num_tokens_per_chunk = math.ceil(
        c2f_model.chunk_size_s * codec_model.sample_rate / codec_model.hop_length
    )
    print(f"{num_tokens_per_chunk=}")
    num_chunks = int(math.ceil(tokens.size(-1) / num_tokens_per_chunk))
    fine_tokens_list = list()
    for chunk_idx in range(num_chunks):
        tmp_tokens = tokens[
            ...,
            chunk_idx * num_tokens_per_chunk : (chunk_idx + 1) * num_tokens_per_chunk,
        ]
        tmp_mask = mask[
            ...,
            chunk_idx * num_tokens_per_chunk : (chunk_idx + 1) * num_tokens_per_chunk,
        ]
        tmp_result = c2f_model.generate(
            codec_model,
            start_tokens=tmp_tokens,
            mask=tmp_mask,
            sampling_steps=24,
            temperature=0.8,
            return_signal=False,
        )
        fine_tokens_list.append(tmp_result)
    fine_tokens = torch.cat(fine_tokens_list, dim=-1)

    # truncate
    fine_tokens = fine_tokens[..., : tokens.size(-1)]
    fine_tokens_path = output_root.joinpath(f"vampnet_{audio_seconds}_fine_tokens")
    np.save(fine_tokens_path, fine_tokens.cpu().numpy())


def decode(input_root):
    device = "cuda:0"
    codec_path = "vampnet_checkpoints/codec.pth"
    codec_model = load_codec(codec_path, device=device)
    path_list = input_root.glob("*.npy")
    for path in path_list:
        with torch.no_grad():
            tmp_tokens = torch.from_numpy(np.load(path)).to(device)
            tmp_embeds = token2embeddings(tmp_tokens, codec_model)
            tmp_dict = codec_model.decode(tmp_embeds)
            tmp_signal = tmp_dict["audio"]
            for idx, sig in enumerate(tmp_signal):
                tmp_sig_path = input_root.joinpath(f"{path.stem}_{idx}.mp3")
                torchaudio.save(
                    tmp_sig_path, sig.cpu(), sample_rate=codec_model.sample_rate
                )
    print("done")


if __name__ == "__main__":
    input_root = Path("vampnet_gen3_30")
    generate(input_root)
    decode(input_root)
