import inspect
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from einops import rearrange, repeat
from typing import Dict, Tuple
import io

from .dit_attention import DiTNetModel
from .pipeline import TrainPipeline
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ModelPipeline(torch.nn.Module):
    def __init__(self,
                 config: Dict,
                 device=torch.device("cuda"),
                 train_batch_size=1,
                 unet_additional_kwargs=None,
                 accelerator=None,
                 ):
        super().__init__()

        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        infer_noise_scheduler_kwargs = config["infer_noise_scheduler_kwargs"]
        self.device = device
        self.train_batch_size = train_batch_size

        motion_module = config['motion_module']
        
        self.unet = DiTNetModel.from_pretrained_2d(config['pretrained_model_path'], subfolder="transformer",
                                                    unet_additional_kwargs=unet_additional_kwargs)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            config['pretrained_model_path'], subfolder="feature_extractor",
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config['pretrained_model_path'], subfolder="image_encoder",
        )
        if 'pretrained_vae_path' in config.keys() and config['pretrained_vae_path'] != "":
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_vae_path'])
        else:
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_model_path'], subfolder="vae")

        unet_tmp_weights = self.unet.pos_embed.proj.weight.clone()
        unet_tmp_bias = self.unet.pos_embed.proj.bias.clone()
        self.unet.pos_embed.proj = torch.nn.Conv2d(
            4 + 4, unet_tmp_weights.shape[0], kernel_size=(2, 2), stride=2, bias=True,
        )
        with torch.no_grad():
            self.unet.pos_embed.proj.weight[:, :4] = unet_tmp_weights # original weights
            self.unet.pos_embed.proj.weight[:, 4:] = torch.zeros(self.unet.pos_embed.proj.weight[:, 4:].shape) # new weights initialized to zero
            self.unet.pos_embed.proj.bias = torch.nn.Parameter(unet_tmp_bias)

        if config.get("enable_xformers_memory_efficient_attention", False):
            self.unet.enable_xformers_memory_efficient_attention()
            accelerator.print("enable_xformers_memory_efficient_attention")

        if config.get("gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
            accelerator.print("enable_gradient_checkpointing")

        if "checkpoint_path" in config.keys() and config['checkpoint_path'] != "":
            checkpoint_path = config['checkpoint_path']
            accelerator.print(f"load all model from checkpoint: {checkpoint_path}")
            checkpoint_path = torch.load(checkpoint_path, map_location="cpu")
            org_state_dict = checkpoint_path["state_dict"] if "state_dict" in checkpoint_path else checkpoint_path        
            
            unet_state_dict = {}

            for name, param in org_state_dict.items():
                if "unet." in name:
                    if name.startswith('module.unet.'):
                        name = name.split('module.unet.')[-1]
                    unet_state_dict[name] = param

            accelerator.print('load checkpoint: unet_state_dict', len(list(unet_state_dict.keys())))

            m, u = self.unet.load_state_dict(unet_state_dict, strict=False)        
            accelerator.print(f"load checkpoint: unet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0, print("unet unexpecting key is", u)

        # 1. unet ckpt
        # 1.1 motion module
        if unet_additional_kwargs['use_motion_module'] and motion_module != "":
            accelerator.print('load motion_module from', motion_module)
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update(
                {"global_step": motion_module_state_dict["global_step"]})
            motion_module_state_dict = motion_module_state_dict[
                'state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
            try:
                # extra steps for self-trained models
                state_dict = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if key.startswith("module."):
                        _key = key.split("module.")[-1]
                        state_dict[_key] = motion_module_state_dict[key]
                    else:
                        state_dict[key] = motion_module_state_dict[key]
                motion_module_state_dict = state_dict
                del state_dict
                accelerator.print(f'load motion_module params len is {len(motion_module_state_dict)}')
                missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
                accelerator.print(f'load motion_module missing {len(missing)}, unexpected {len(unexpected)}')
                assert len(unexpected) == 0
            except:
                _tmp_ = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if "motion_modules" in key:
                        if key.startswith("unet."):
                            _key = key.split('unet.')[-1]
                            _tmp_[_key] = motion_module_state_dict[key]
                        else:
                            _tmp_[key] = motion_module_state_dict[key]
                accelerator.print(f'load motion_module params len is {len(_tmp_)}')            
                missing, unexpected = self.unet.load_state_dict(_tmp_, strict=False)
                accelerator.print(f'load motion_module missing {len(missing)}, unexpected {len(unexpected)}')
                assert len(unexpected) == 0
                del _tmp_
            del motion_module_state_dict

        
        self.vae.to(device=self.device, dtype=torch.float16)
        self.unet.to(device=self.device, dtype=torch.float16)
        self.image_encoder.to(device=self.device, dtype=torch.float16)

        scheduler = DPMSolverMultistepScheduler(**OmegaConf.to_container(infer_noise_scheduler_kwargs))
        self.pipeline = TrainPipeline(
            vae=self.vae, unet=self.unet,
            clip_image_processor=self.clip_image_processor,
            image_encoder=self.image_encoder,
            scheduler=scheduler,
            # NOTE: UniPCMultistepScheduler
        )
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

    def infer(self, 
            source_image, 
            prompt_embeddings,
            pose_image,
            random_seed, 
            step, 
            guidance_scale, 
            context,
            accelerator,
            size=(512, 768),
            video_length=64,
            prompt = "",
            n_prompt = "",
            alpha = 1,
            block_size = 32, # For 1024x1024 image, meaning that a grid of 32 × 32 squares 
            ):

        step = int(step)

        set_seed(random_seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        B, C, H, W = source_image.shape

        init_latents = None
        
        context_frames = context["context_frames"]
        context_stride = context["context_stride"]
        context_overlap = context["context_overlap"]
        return self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            prompt_embeddings=prompt_embeddings,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            pose_image=(pose_image + 1.) / 2., # become [0, 1]
            init_latents=init_latents,
            generator=generator,
            source_image=source_image,
            context_frames = context_frames,
            context_stride = context_stride,
            context_overlap = context_overlap,
            size = size,
            video_length = video_length,
            accelerator = accelerator,
            alpha = alpha,
            block_size = block_size,
        ).videos

    def forward(self,
                init_latents,
                prompt_embeddings,
                timestep,
                source_image,
                motion_sequence,
                clip_image,
                guidance_scale,
                prompt = "",
                n_prompt = "",
                ):

        control = (motion_sequence + 1.) / 2. # become [0, 1]
        B, C, H, W  = source_image.shape
        noise_pred = self.pipeline.train(
            prompt,
            prompt_embeddings=prompt_embeddings,
            negative_prompt=n_prompt,
            timestep=timestep,
            width=W,
            height=H,
            video_length=control.shape[1],
            controlnet_condition=control,
            init_latents=init_latents,  # add noise to latents
            source_image=source_image,
            clip_image=clip_image,
        )
        return noise_pred