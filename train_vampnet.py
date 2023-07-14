from pathlib import Path
import json
import time
import datetime
import typing
import argparse


# torch
import accelerate
import torch
import einops
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf

from audiocraft.models.loaders import load_compression_model
from audiocraft.models.encodec import EncodecModel

import vampnet
from vampnet.modules.transformer import VampNet
from vampnet import mask as vampnet_mask
from vampnet import util as vampnet_util

# dataset
from datasets import PreprocessedMusicLmDataset

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


def get_config(path: str) -> dict:
    with open(path, mode="r") as rf:
        result = json.load(rf)
    return result


def load_encodec(file_or_url: str = None, device="cpu"):
    if not file_or_url:
        file_or_url = "https://dl.fbaipublicfiles.com/audiocraft/musicgen/v0/b0dbef54-37d256b525.th"
    model = load_compression_model(file_or_url).to(device)
    return model


def save_checkpoint(
    save_root: Path,
    prefix: str,
    steps: int,
    model: VampNetWrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    def _save(model: torch.nn.Module, model_path: Path):
        torch.save(model.state_dict(), model_path)
        print(f"{datetime.datetime.now()} done saving {model_path.as_posix()}")

    for m, n in zip((model, optimizer, scheduler), ("model", "optimizer", "scheduler")):
        p = save_root.joinpath(f"{prefix}_{n}_{steps}.pth")
        _save(m, p)
    return


def load_checkpoint(
    checkpoint_config: dict,
    model: VampNetWrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
):
    def _load(model, model_path, device):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"{datetime.datetime.now()} done loading {model_path}")

    if not checkpoint_config:
        return None
    model_path = checkpoint_config.get("model_path", None)
    if model_path:
        _load(model, model_path, device)
    optimizer_path = checkpoint_config.get("optimizer_path", None)
    if optimizer_path:
        _load(optimizer, optimizer_path, device)
    scheduler_path = checkpoint_config.get("scheduler_path", None)
    if scheduler_path:
        _load(scheduler, scheduler_path, device)
    return


def get_accuracy(
    preds: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 1,
    ignore_index: typing.Optional[int] = None,
) -> torch.Tensor:
    # ThomasLee
    # preds: [batch_size, logits, C1, C2, ...]
    # target: [batch_size, C1, C2, ...]
    # ThomasLee: check input validity
    if target.numel() < 1:
        print("empty target, return nan.")
        return torch.tensor(float("nan"), device=target.device)

    # Flatten the predictions and targets to be of shape (batch_size * sequence_length, n_class)
    # preds = einops.rearrange(preds, "b p s -> (b s) p")
    # target = einops.rearrange(target, "b s -> (b s)")

    # return torchmetrics.functional.accuracy(preds, target, task='multiclass', top_k=topk, num_classes=preds.shape[-1], ignore_index=ignore_index)
    # if ignore_index is not None:
    #     # Create a mask for the ignored index
    #     mask = target != ignore_index
    #     # Apply the mask to the target and predictions
    #     preds = preds[mask.unsqueeze(1)]
    #     target = target[mask]

    # reshape target to [batch_size, 1, C1, C2, ...]
    target = target.unsqueeze(1)

    # Get the top-k predicted classes and their indices
    _, pred_indices = torch.topk(preds, k=top_k, dim=1)

    # Determine if the true target is in the top-k predicted classes
    correct = torch.eq(pred_indices, target)
    # apply mask
    if ignore_index is None:
        mask = torch.ones_like(target, device=target.device)
    else:
        mask = target != ignore_index
    correct *= mask

    # Calculate the accuracy
    accuracy = torch.sum(correct) / torch.sum(mask)

    return accuracy


def get_metrics(logits, target, sampled_dist, target_mask, prefix: str = "training"):
    output = dict()
    for dist_range in [(0, 0.5), (0.5, 1.0)]:
        unmasked_target = target.masked_fill(target_mask.bool(), IGNORE_INDEX)
        masked_target = target.masked_fill(~target_mask.bool(), IGNORE_INDEX)

        assert target.shape[0] == sampled_dist.shape[0]
        # grab the indices of the r values that are in the range
        sampled_idx = (sampled_dist >= dist_range[0]) & (sampled_dist < dist_range[1])

        # grab the target and logits values that are in the range
        r_unmasked_target = unmasked_target[sampled_idx]
        r_masked_target = masked_target[sampled_idx]
        r_logits = logits[sampled_idx]

        for topk in (1, 25):
            s, e = dist_range
            tag = f"{prefix}-{s}-{e}/top{topk}"

            output[f"{tag}/unmasked"] = get_accuracy(
                preds=r_logits,
                target=r_unmasked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )
            output[f"{tag}/masked"] = get_accuracy(
                preds=r_logits,
                target=r_masked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )
    return output


def get_preprocessed_loss_and_accuracy(
    batch_data: torch.Tensor,
    accelerator: accelerate.Accelerator,
    vampnet_wrapper: VampNetWrapper,
    encodec_weight_list: typing.List[torch.Tensor],
    rng: torch.quasirandom.SobolEngine,
    metric_prefix: str,
):
    unwrapped_model = accelerator.unwrap_model(vampnet_wrapper)
    prompt_tokens = batch_data
    prompt_tokens = einops.rearrange(prompt_tokens, "b s q -> b q s").contiguous()
    device = prompt_tokens.device
    batch_size = prompt_tokens.size(0)
    # sample layer_idx and generate mask, where tokens become [MASK] where mask==1
    sampled_dist = rng.draw(batch_size)[:, 0].to(device)
    sampled_mask = vampnet_mask.random(prompt_tokens, sampled_dist)
    sampled_mask = vampnet_mask.codebook_unmask(
        sampled_mask, unwrapped_model.model.n_conditioning_codebooks
    )
    masked_prompt_tokens, sampled_mask = vampnet_mask.apply_mask(
        prompt_tokens, sampled_mask, unwrapped_model.model.mask_token
    )
    # token to embedding
    prompt_embeddings = unwrapped_model.model.embedding.from_codes2(
        masked_prompt_tokens, encodec_weight_list
    )
    logits = vampnet_wrapper(prompt_embeddings, sampled_dist)
    # generate target
    target = prompt_tokens[:, unwrapped_model.model.n_conditioning_codebooks :, :]
    target_mask = sampled_mask[:, unwrapped_model.model.n_conditioning_codebooks :, :]
    masked_target = target.masked_fill(~target_mask.bool(), IGNORE_INDEX)
    loss = torch.nn.functional.cross_entropy(
        logits, masked_target, label_smoothing=0.1, ignore_index=IGNORE_INDEX
    )
    # metrics
    metric_dict = get_metrics(
        logits, target, sampled_dist, target_mask, prefix=metric_prefix
    )
    return loss, metric_dict


def gather_and_mean(local_metric_dict: dict, accelerator: accelerate.Accelerator):
    result = dict()
    for k, v in local_metric_dict.items():
        gathered_v = accelerator.gather(v)
        mean_v = gathered_v.mean().item()
        result[k] = mean_v
    return result


def train(args):
    """multi-node multi-gpu"""

    # config
    config_dict = get_config(args.config_path)
    training_config: dict = config_dict["training_config"]
    checkpoint_config: dict = training_config.get("checkpoint_config", None)
    model_dir = Path(training_config["model_dir"])
    assert model_dir.exists() and model_dir.is_dir(), f"invalid {model_dir=}"
    log_dir = training_config["summarywriter_root"]
    training_seed = time.time()
    if training_config["constant_seed"]:
        training_seed = 0
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)

    # rng
    # a better rng for sampling from our schedule
    rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=round(training_seed))

    # accelerator
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
    accelerator_kwargs = {
        "log_with": "tensorboard",
        "project_dir": log_dir,
        "split_batches": False,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    rank = accelerator.process_index
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes
    device = accelerator.device
    accelerator.init_trackers(f"unconditional_musicgen_{datetime.datetime.now()}")
    print(f"{rank=} {local_rank=} {world_size=} {device=}")

    # init dataset
    dataset_config = config_dict["dataset_config"]
    dataset = PreprocessedMusicLmDataset(**dataset_config.get("config"))
    dataset_training, dataset_eval = random_split(
        dataset, [0.99, 0.01], generator=torch.Generator().manual_seed(0)
    )
    accelerator.print(
        f"{datetime.datetime.now()}: training_dataset={len(dataset_training)} eval_dataset={len(dataset_eval)} total={len(dataset_training) + len(dataset_eval)}"
    )
    # init dataloader
    dataloader_kwargs = {
        "batch_size": training_config["batch_size"],
        "shuffle": True,
        "pin_memory": True,
        "persistent_workers": True,
        "num_workers": 4,
        # "multiprocessing_context": "forkserver",
    }
    dataloader_training = DataLoader(dataset_training, **dataloader_kwargs)
    dataloader_eval = DataLoader(dataset_eval, **dataloader_kwargs)

    # init encodec
    encodec_model: EncodecModel = load_encodec(device=device)
    # encodec_weight should be [num_tokens, dim]
    encodec_weight_list = [
        encodec_model.quantizer.vq.layers[i]._codebook.embed
        for i in range(len(encodec_model.quantizer.vq.layers))
    ]

    # init model
    vampnet_config_path = config_dict["vampnet"]["config_path"]
    accelerator.print(f"using {vampnet_config_path=}")
    vampnet_wrapper = VampNetWrapper(vampnet_config_path).to(device)
    accelerator.print(vampnet_wrapper.model)

    # optimizer
    max_grad_norms = 1.0
    opt_config = {"lr": 0.001, "betas": [0.9, 0.95], "weight_decay": 0.1}
    optimizer = torch.optim.AdamW(vampnet_wrapper.parameters(), **opt_config)
    # scheduler
    scheduler_config = {"d_model": vampnet_wrapper.model.embedding_dim}
    scheduler = vampnet.scheduler.NoamScheduler(optimizer, **scheduler_config)
    # load checkpoint
    load_checkpoint(checkpoint_config, vampnet_wrapper, optimizer, scheduler, device)
    # prepare
    (
        vampnet_wrapper,
        optimizer,
        scheduler,
        dataloader_training,
        dataloader_eval,
    ) = accelerator.prepare(
        vampnet_wrapper, optimizer, scheduler, dataloader_training, dataloader_eval
    )
    # load step
    global_step = 0
    if checkpoint_config:
        global_step = checkpoint_config.get("global_step")
        accelerator.print(f"continue from {global_step=}")
    # train loop
    while (
        global_step
        < training_config["num_epochs"] * len(dataloader_training) + global_step
    ):
        if rank == 0:
            epoch_time = time.time()
        for step_idx, batch_data in enumerate(dataloader_training, start=1):
            grad_norms = None
            learning_rate = None
            gathered_training_metric_dict = dict()
            gathered_eval_metric_dict = dict()
            vampnet_wrapper.train()
            with accelerator.accumulate(vampnet_wrapper):
                (
                    training_loss,
                    training_metric_dict,
                ) = get_preprocessed_loss_and_accuracy(
                    batch_data,
                    accelerator,
                    vampnet_wrapper,
                    encodec_weight_list,
                    rng,
                    "training",
                )
                # backward
                accelerator.backward(training_loss)
                # gradient clipping
                if accelerator.sync_gradients:
                    grad_norms = accelerator.clip_grad_norm_(
                        vampnet_wrapper.parameters(), max_grad_norms
                    ).item()
                    # update only when gradients are synced
                    scheduler.step()
                # update and gather
                training_metric_dict.update(training_loss=training_loss)
                gathered_training_metric_dict = gather_and_mean(
                    training_metric_dict, accelerator
                )
                optimizer.step()
                # NOTE: get_lr() returns a list of learning rates
                learning_rate = scheduler.get_lr()
                optimizer.zero_grad()
                # update and sync
                global_step += 1
                if accelerator.sync_gradients:
                    step_list = [None]
                    if rank == 0:
                        step_list[0] = global_step
                    step_list = accelerate.utils.broadcast_object_list(
                        step_list, from_process=0
                    )
                    global_step = step_list[0]
                    accelerator.print(
                        f"{datetime.datetime.now()}: {rank=} {local_rank=} {global_step=} {step_idx=} {grad_norms=} {gathered_training_metric_dict=}"
                    )
            # save models
            if accelerator.is_main_process and not (
                global_step % training_config["save_steps"]
            ):
                save_checkpoint(
                    model_dir,
                    "unconditional_vampnet",
                    global_step,
                    accelerator.unwrap_model(vampnet_wrapper),
                    optimizer,
                    scheduler,
                )
            # evaluate
            if not (global_step % training_config["eval_steps"]):
                vampnet_wrapper.eval()
                with torch.no_grad():
                    for eval_step_idx, eval_batch_data in enumerate(
                        dataloader_eval, start=1
                    ):
                        (
                            eval_loss,
                            eval_metric_dict,
                        ) = get_preprocessed_loss_and_accuracy(
                            eval_batch_data,
                            accelerator,
                            vampnet_wrapper,
                            encodec_weight_list,
                            rng,
                            "eval",
                        )
                        # TODO: more eval steps
                        break
                    # update and gather
                    eval_metric_dict.update(eval_loss=eval_loss)
                    gathered_eval_metric_dict = gather_and_mean(
                        eval_metric_dict, accelerator
                    )
                    accelerator.print(
                        f"{datetime.datetime.now()} {rank=} {local_rank=} {global_step=} {gathered_eval_metric_dict=}"
                    )
            # log
            accelerator.log(
                {
                    "grad_norms": grad_norms,
                    "learning_rate": learning_rate[0],
                    **gathered_training_metric_dict,
                    **gathered_eval_metric_dict,
                },
                step=global_step,
            )
        # get epoch time
        if rank == 0:
            print(
                f"{datetime.datetime.now()} epoch_time={time.time() - epoch_time:.4f}s"
            )
        accelerator.wait_for_everyone()
    print(f"{datetime.datetime.now()}: done")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_path", type=str, default=None, help="path to config.json"
    )
    args = arg_parser.parse_args()
    train(args)
