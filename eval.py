import argparse
import datetime
import inspect
import os
import random
import numpy as np
import cv2

from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm

from accelerate.utils import set_seed
from einops import rearrange

from pathlib import Path
import io
from accelerate import Accelerator

def eval_model(validation_data,
                 model, 
                 local_rank, 
                 weight_type, 
                 context,
                 output_dir,
                 global_step,
                 accelerator,
                 valid_seed,
                 ):
    sample_size = validation_data['sample_size']
    guidance_scale = validation_data['guidance_scale']
    video_length = validation_data['video_length']
    Image.MAX_IMAGE_PIXELS = None

    # input test videos (either source video/ conditions)

    pose_images = validation_data['pose_image']
    source_images = validation_data['source_image']
    target_images = validation_data['target_image']

    # read size, step from yaml file
    ori_model = accelerator.unwrap_model(model)

    for idx, (source_image, pose_image, target_image) in tqdm(
        enumerate(zip(source_images, pose_images, target_images)),
        total=len(pose_images), disable=(not accelerator.is_main_process)
    ):
        # 从左往右的保存结果分别为IHC的源图，HE的源图，HE的生成结果
        source_image = Image.open(source_image).convert("RGB")
        source_image = np.array(source_image)
        source_image = cv2.GaussianBlur(cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR), (51, 51), sigmaX=0, sigmaY=0)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        
        source_image = torch.tensor(np.array(source_image)).unsqueeze(0).to(local_rank, dtype=weight_type)
        source_image = rearrange(source_image, "b h w c -> b c h w") # b c h w
        
        pose_image = Image.open(pose_image).convert("RGB")
        pose_image = torch.tensor(np.array(pose_image)).unsqueeze(0).to(dtype=weight_type)
        pose_image = rearrange(pose_image, "b h w c -> b c h w") # b c h w
        
        image_prompt_embeddings = None
        with torch.no_grad():
            samples = ori_model.infer(
                source_image=source_image / 127.5 - 1.,
                prompt_embeddings=image_prompt_embeddings,
                pose_image=pose_image / 127.5 - 1.,
                step=validation_data['num_inference_steps'],
                guidance_scale=guidance_scale,
                context=context,
                size=sample_size,
                video_length=video_length,
                random_seed=valid_seed,
                accelerator=accelerator,
                alpha=validation_data.get("alpha", 1),
                block_size=validation_data.get("block_size", 32),
            )
            
            source_image = torch.nn.functional.interpolate(source_image, size=(pose_image.shape[2], pose_image.shape[3]))
            samples = samples.cpu() * 255.
            samples = rearrange(samples[0], "c h w -> h w c")
            res_h, res_w = samples.shape[:2]
            accelerator.print(f"{res_h=}, {res_w=}")
        if accelerator.is_main_process:
            os.makedirs(f"{output_dir}/samples/sample_{global_step}", exist_ok=True)
            pose_name = os.path.basename(validation_data['pose_image'][idx]).split(".")[0]
            source_name = os.path.basename(validation_data['source_image'][idx]).split(".")[0]
            time_prob = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            save_path = f"{output_dir}/samples/sample_{global_step}/{source_name}_{pose_name}-{time_prob}.png"
            os.makedirs(f"{output_dir}/samples/sample_{global_step}", exist_ok=True)
            Image.fromarray(samples.cpu().numpy().clip(0, 255).astype("uint8")).save(save_path)
            accelerator.print(f"Saved samples to {save_path}")
        # torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    return f"{output_dir}/samples/sample_{global_step}"

def main(args):

    weight_type = torch.float16
    # Accelerate
    accelerator = Accelerator()
    local_rank = accelerator.device
    config = OmegaConf.load(args.config)
    # Initialize distributed training
    savedir = None
    if accelerator.is_main_process:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"DVST_samples/{Path(args.config).stem}-{time_str}"
        os.makedirs(savedir, exist_ok=True)
        OmegaConf.save(config, f"{savedir}/config.yaml")
    accelerator.wait_for_everyone()
    # inference_config = OmegaConf.load(config.inference_config)
    model_type = config.get(
        "model_type", "pixart_small"
    )

    if model_type in ["dvst_modules"]:
        from models.dvst_modules.model import ModelPipeline

    model = ModelPipeline(config=config,
                            train_batch_size=1,
                            device=local_rank,
                            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs),
                            accelerator=accelerator,)
    
    model.eval()
    eval_model(
        validation_data = config.validation_data, 
        model = model, 
        local_rank = local_rank, 
        weight_type = weight_type, 
        context = config.context, 
        output_dir = savedir, 
        global_step = 0,
        accelerator = accelerator,
        valid_seed = config.valid_seed,
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--condition", type=str, required=False, default="control_eye_mouths")# select "control_eye_mouths", "control_eyes", "zero"
    args = parser.parse_args()
    main(args)
