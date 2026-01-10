import os
import math
import random
import logging
import inspect
import argparse
import datetime
import threading
import subprocess
import cv2

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from typing import Dict, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from diffusers.optimization import get_scheduler
from accelerate import Accelerator

from eval import eval_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def main(
        image_finetune: bool,

        origin_config,
        name: str,
        output_dir: str,
        checkpoint_output_dir: str,

        train_data: Dict,
        validation_data: Dict,
        context: Dict,

        pretrained_model_path: str = "",
        pretrained_vae_path: str = "",
        empty_str_embedding: str = "",
        motion_module: str = "",
        checkpoint_path: str = "",
        unet_additional_kwargs: Dict = {},
        noise_scheduler_kwargs=None,
        infer_noise_scheduler_kwargs=None,

        max_train_epoch: int = -1,
        validation_steps: int = 100,
        validation_steps_tuple: Tuple = (-1,),

        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",

        trainable_modules: Tuple[str] = (None,),
        num_workers: int = 8,
        train_batch_size: int = 1,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        checkpointing_steps: int = -1,

        mixed_precision_training: bool = True,
        enable_xformers_memory_efficient_attention: bool = True,

        valid_seed: int = 42,
        is_debug: bool = False,

        model_type: str = "unet",
        appearance_time_step: int = 0,
        resume_step_offset: int = 0,
        recycle_seed: int = 100000000
):

    # check params is true to run
    assert validation_steps % gradient_accumulation_steps == 0, "Error, gradient_accumulation_steps must be divided by validation_steps"
    assert checkpointing_steps % gradient_accumulation_steps == 0, "Error, gradient_accumulation_steps must be divided by checkpointing_steps"
    weight_type = torch.float16
    # Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    set_seed(accelerator.process_index)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    checkpoint_output_dir = os.path.join(checkpoint_output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    if accelerator.state.deepspeed_plugin is not None and \
            accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto":
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_batch_size

    # Load tokenizer and models.
    
    local_rank = accelerator.device

    if model_type in ["dvst_modules"]:
        from models.dvst_modules.model import ModelPipeline
        from models.dvst_modules.dataset import VideosIterableDataset


    from models.variational_lower_bound_loss import IDDPM
    vlb_loss_func = IDDPM(
        str(1000), learn_sigma=True, pred_sigma=True, snr=False
    )
    accelerator.print("using VariationalLowerBoundLoss and mse_loss")
    def train_loss_func(model_output,
                        x_t,
                        x_start,
                        target,
                        timestep,
                        ):
        return vlb_loss_func(model_output, x_t, x_start, timestep, {}, target)

    model = ModelPipeline(config=config,
                         train_batch_size=train_batch_size,
                         device=local_rank,
                         unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
                         accelerator=accelerator,)
    vae_scaling_factor = model.vae.config.scaling_factor
    # Load noise_scheduler
    noise_scheduler = model.scheduler
    # ----- load image encoder ----- #

    # Set trainable parameters
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    accelerator.print('trainable_params', len(trainable_params))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    accelerator.print(f"trainable params number: {len(trainable_params)}")
    accelerator.print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    model.to(local_rank)
    
    video_length = train_data["video_length"]
    resolution = train_data['resolution']
    train_dataset = VideosIterableDataset(
        data_dir_pre        = train_data['data_dir_pre'],
        resolution          = resolution,
        group_size          = train_data['group_size'],
        img_resolution      = train_data['img_resolution'],
        video_length        = video_length,
        pose_dir_name       = train_data["pose_dir"],
        target_dir_name     = train_data["target_dir"],
        source_from_self    = train_data.get("source_from_self", False),
        blur_ref_image      = train_data.get("blur_ref_image", False),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # save train config
    if accelerator.is_main_process:
        OmegaConf.save(origin_config, f"{output_dir}/config.yaml")
        OmegaConf.save(origin_config, f"{checkpoint_output_dir}/config.yaml")

    max_train_steps = len(train_dataset) * max_train_epoch
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {max_train_epoch}")
    accelerator.print(f"  Instantaneous batch size per device = {train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {max_train_steps}")
    accelerator.print(f"  num_processes = {accelerator.num_processes}")
    global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training

    model, \
    optimizer, \
    lr_scheduler, \
    train_dataloader = accelerator.prepare(model, 
                                            optimizer, 
                                            lr_scheduler,
                                            train_dataloader
                                            )
    for epoch in range(0, max_train_epoch):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                # Get video data as origin       
                pixel_values = batch["pixel_values"].to(local_rank, dtype=weight_type) # b f c h w
                # Chose ref-image
                pixel_values_ref_img = batch["pixel_values_ref_img"].to(local_rank, dtype=weight_type) # b c h w
                pixel_values_pose = batch["pixel_values_pose"].to(local_rank, dtype=weight_type)# b f c h w
                clip_image = batch["clip_image"].to(local_rank, dtype=weight_type)# b c h w
                
                # Get ref-image and Pose Condition to PoseGuider
                # pixel_values_ref_img is b c h w with value is 0-255
                pixel_values_ref_img = pixel_values_ref_img / 127.5 - 1.
                pixel_values_pose = pixel_values_pose / 127.5 - 1.
                
                # NOTE: convert pixel_values(origin video) to latent by vae
                pixel_values = pixel_values / 127.5 - 1
                with torch.no_grad():
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = model.module.vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", b=train_batch_size)
                    latents = latents * vae_scaling_factor
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                noise = noise + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
                bsz = latents.shape[0]

                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = vlb_loss_func.q_sample(latents, timesteps, noise=noise).to(local_rank, dtype=weight_type)
                # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                with accelerator.autocast():
                    model_pred = model(init_latents=noisy_latents,
                                    prompt_embeddings=None,
                                    timestep=timesteps,
                                    guidance_scale=1.0,
                                    source_image=pixel_values_ref_img, 
                                    motion_sequence=pixel_values_pose,
                                    clip_image=clip_image,
                                    )
                    loss = train_loss_func(
                        model_output = model_pred.float(),
                        x_t = noisy_latents.float(),
                        x_start = latents.float(),
                        target = target.float(),
                        timestep = timesteps.to(device=local_rank),
                    )
                    avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                    train_loss += avg_loss.item() / gradient_accumulation_steps
                    # use accelerator
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            set_seed(global_step + accelerator.process_index)
            
            # Save checkpoint
            if global_step in validation_steps_tuple or global_step % validation_steps == 0:
                accelerator.wait_for_everyone()
                # torch.cuda.empty_cache()
                try:
                    infer_sample_path = eval_model(validation_data, 
                            model, 
                            local_rank, 
                            weight_type, 
                            context, 
                            output_dir, 
                            global_step,
                            accelerator,
                            valid_seed,
                            )
                    if accelerator.is_main_process:
                        os.system(f"cp -r {infer_sample_path} {checkpoint_output_dir}")
                except Exception as e:
                    import traceback; traceback.print_exc()
                # torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                
            if global_step % checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = checkpoint_output_dir
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                    }
                    
                    model_save_path = os.path.join(save_path, f"checkpoint-steps{global_step}.ckpt")
                    torch.save(state_dict, model_save_path)
                accelerator.wait_for_everyone()

            logs = {"step_loss": avg_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, origin_config=config, **config)
