import inspect
import math
import random
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.distributed as dist
from tqdm import tqdm
from diffusers.utils import is_accelerate_available
from packaging import version
from accelerate import Accelerator
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from .dit_attention import DiTNetModel
from ..context import (
    get_context_scheduler,
    get_total_steps
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

# two input source_image is (-1, 1)
# control is (0, 1)

class TrainPipeline(DiffusionPipeline):
    _optional_components = []
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: DiTNetModel,
        clip_image_processor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            # deprecation_message = (
            #     f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
            #     f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
            #     "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
            #     " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
            #     " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
            #     " file"
            # )
            # deprecate("steps_offset!=1", "1.0.0",
            #           deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0",
                      deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(
            unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0",
                      deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        self.negative_prompt_embeds = None
        self.negative_prompt_attention_mask = None
        self.register_modules(
            vae=vae,
            unet=unet,
            clip_image_processor=clip_image_processor,
            image_encoder=image_encoder,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.vae.enable_tiling()
        self.vae.tile_overlap_factor = 0.25
        # self.vae.enable_slicing()
        
        # self.negative_prompt_embeds = self.negative_prompt_embeds.to(dtype=torch.float16, device=self._execution_device)
        # self.negative_prompt_attention_mask = self.negative_prompt_attention_mask.to(dtype=torch.float16, device=self._execution_device)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, 
                        height, width, dtype, device, generator,):
        shape = (
            batch_size, num_channels_latents, height // self.vae_scale_factor,
            width // self.vae_scale_factor)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(
                    shape, generator=generator, device=rand_device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def next_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps //
                       self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def split_video_format(self, latents, patch_h, patch_w, view_list):
        patches = []
        for h_start, w_start in view_list:
            patch = latents[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w]
            patches.append(patch)
        return torch.stack(patches, dim=2)
    
    def merge_video_format(self, latents, height, width, patch_h, patch_w, view_list):
        b, c, f, h, w = latents.shape
        merged = torch.zeros((b, c, height, width), device=latents.device, dtype=latents.dtype)
        count = torch.zeros((b, c, height, width), device=latents.device, dtype=latents.dtype)
        for idx, (h_start, w_start) in enumerate(view_list):
            merged[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] += latents[:, :, idx]
            count[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] += 1
        return merged / count
    
    def get_view(self, height, width, patch_h, patch_w, number, timestep, probabilities):
        number_h = height // patch_h + (height % patch_h != 0)
        number_w = width // patch_w + (width % patch_w != 0)
        assert number >= number_h * number_w, "number of patch must make All Patches cover full image"
        view_list = []
        # Get full cover patch
        for i in range(number_h):
            for j in range(number_w):
                h_start = i * patch_h
                if h_start + patch_h > height:
                    h_start = height - patch_h
                w_start = j * patch_w
                if w_start + patch_w > width:
                    w_start = width - patch_w
                view_list.append((h_start, w_start))
        # print("view_list is", view_list)
        number -= number_h * number_w
        rng = np.random.default_rng(timestep + 42)
        list_select = []
        list_prob = []
        for i in range(0, height - patch_h + 1):
            for j in range(0, width - patch_w + 1):
                list_select.append((i, j))
                list_prob.append(probabilities[i + patch_h // 2, j + patch_w // 2])
        list_prob = np.array(list_prob)
        list_prob = list_prob / list_prob.sum()    
        list_select = np.array(list_select)  # shape: (N, 2)
        number = min(number, len(list_select))  # 防止溢出
        chosen_indices = rng.choice(len(list_select), size=number, replace=False, p=list_prob)
        for idx in chosen_indices:
            chosen_row, chosen_col = list_select[idx]
            if chosen_row + patch_h > height:
                chosen_row = height - patch_h
            if chosen_col + patch_w > width:
                chosen_col = width - patch_w
            view_list.append((chosen_row, chosen_col))

        return view_list
    
    def get_probabilities(self, imgs, alpha=1, block_size=32):
        img = rearrange(imgs[0] * 255, "c h w -> h w c").cpu().numpy().astype("uint8")
        img = np.array(Image.fromarray(img).convert("L"))
        height, width = img.shape

        # 计算块的行列数
        rows = height // block_size
        cols = width // block_size

        combined_heatmap = np.zeros((rows * block_size, cols * block_size))

        for i in range(rows):
            for j in range(cols):
                block = img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                f_transform = np.fft.fft2(block)
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_transform_shifted)
                magnitude_log = np.log1p(magnitude)
                combined_heatmap[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = magnitude_log.mean()
        combined_heatmap = zoom(combined_heatmap, (1 / self.vae_scale_factor, 1 / self.vae_scale_factor), order=1)
        combined_heatmap = np.max(combined_heatmap) + np.min(combined_heatmap) - combined_heatmap
        combined_heatmap = combined_heatmap ** alpha
        probabilities = combined_heatmap / combined_heatmap.sum()
        return probabilities

    
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            accelerator: Accelerator,
            prompt_embeddings: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: List[float] = [7.5],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator,
                                      List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            pose_image: Optional[torch.FloatTensor] = None,
            context_frames: int = 16,
            context_stride: int = 1,
            context_overlap: int = 4,
            context_batch_size: int = 1,
            context_schedule: str = "uniform",
            size: tuple = (512, 512),
            video_length: int = 64,
            init_latents: Optional[torch.FloatTensor] = None,
            num_actual_inference_steps: Optional[int] = None,
            appearance_encoder=None,
            reference_control_writer=None,
            reference_control_reader=None,
            source_image: torch.FloatTensor = None,
            decoder_consistency=None,
            alpha = 1,
            block_size = 32,
            **kwargs,
    ):
        
        height ,width = size[0] // self.vae_scale_factor, size[1] // self.vae_scale_factor
        video_length = video_length * (pose_image.shape[2] // size[0]) * (pose_image.shape[3] // size[1])
        accelerator.print(f"{size=}, {pose_image.shape=}, {source_image.shape=}, {video_length=}")
        device = source_image.device
        dtype = source_image.dtype
        do_classifier_free_guidance = True if sum(guidance_scale) > len(guidance_scale) else False
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        rank = accelerator.process_index
        world_size = accelerator.num_processes

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            num_videos_per_prompt,
            num_channels_latents,
            height = pose_image.shape[2],
            width = pose_image.shape[3],
            dtype = dtype,
            device = device,
            generator = generator,
        ) 
        source_image = torch.nn.functional.interpolate(source_image, (height, width))
        probabilities = self.get_probabilities(pose_image, alpha, block_size)
        latent_channels = latents.shape[1]
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps
        
        context_scheduler = get_context_scheduler(context_schedule)
        
        clip_source_img = rearrange(source_image, "b c h w -> b h w c")[0].cpu().numpy().astype(np.float32)
        clip_image = self.clip_image_processor.preprocess((clip_source_img / 2. + 0.5),
                    return_tensors="pt",
                    do_rescale=False).pixel_values # b c h w
        prompt_embeds = self.image_encoder(clip_image.to(device, dtype=dtype), 
                                           return_dict=True).last_hidden_state
        accelerator.print(f"{prompt_embeds.shape=}")

        for t_i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=(not accelerator.is_main_process)):
            if num_actual_inference_steps is not None and t_i < num_inference_steps - num_actual_inference_steps:
                continue
            view_list = self.get_view(height=pose_image.shape[2] // self.vae_scale_factor, 
                                      width=pose_image.shape[3] // self.vae_scale_factor, 
                                      patch_h=height, 
                                      patch_w=width,
                                      number=video_length,
                                      timestep=0,
                                      probabilities=probabilities,
                                      )
            # view_list in vae scale !!!
            latents = self.split_video_format(latents, height, width, view_list)
            mul_view_list = [(x * self.vae_scale_factor, y * self.vae_scale_factor) for x, y in view_list]
            controlnet_cond_images = self.split_video_format(pose_image, height * self.vae_scale_factor, width * self.vae_scale_factor, mul_view_list)

            noise_pred = torch.zeros(
                (latents.shape[0] * (3 if do_classifier_free_guidance else 1),
                 *latents.shape[1:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            counter = torch.zeros(
                (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
            )

            context_queue = list(context_scheduler(
                0, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
            ))

            num_context_batches = math.ceil(
                len(context_queue) / context_batch_size)
            global_context = []
            for i in range(num_context_batches):
                old_context = context_queue[i * context_batch_size: (i + 1) * context_batch_size]
                context = [[0 for _ in range(len(old_context[c_j]))] for c_j in range(len(old_context))]
                for c_j in range(len(old_context)):
                    for c_i in range(len(old_context[c_j])):
                        context[c_j][c_i] = (old_context[c_j][c_i] + t_i * 0) % video_length
                global_context.append(context)
                
            if rank < len(global_context):
                for context in tqdm(global_context[rank::world_size]):
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(3 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )

                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t)

                    pose_guide_conditions=torch.cat(
                            [controlnet_cond_images[:, :, c] for c in context], dim=2)

                    pose_guide_conditions = torch.cat([torch.zeros_like(pose_guide_conditions), 
                                                        pose_guide_conditions, 
                                                        pose_guide_conditions]) \
                        if do_classifier_free_guidance else pose_guide_conditions
                    encoder_hidden_states = torch.cat([torch.zeros_like(prompt_embeds), torch.zeros_like(prompt_embeds), prompt_embeds]) \
                        if do_classifier_free_guidance else prompt_embeds
                    pose_guide_conditions = pose_guide_conditions.to(device=device, dtype=dtype)
                    pred = self.unet(
                        latent_model_input, 
                        timestep=t, 
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=None,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                        pose_guide_conditions=pose_guide_conditions,
                    )[0]

                    if do_classifier_free_guidance:
                        pred_uc_0, pred_uc_1, pred_c = pred.chunk(3)
                        pred = torch.cat([pred_uc_0.unsqueeze(0), pred_uc_1.unsqueeze(0), pred_c.unsqueeze(0)])
                    else:
                        pred = pred.unsqueeze(1)
                    pred, _ = torch.split(pred, latent_channels, dim=2)
                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred[:, j]
                        counter[:, :, c] = counter[:, :, c] + 1

            noise_pred_gathered = [torch.zeros_like(
                noise_pred) for _ in range(world_size)]
            noise_pred_gathered = accelerator.gather(tensor=noise_pred)
            noise_pred_gathered = rearrange(noise_pred_gathered, "(n b) c f h w -> n b c f h w", b=3 if do_classifier_free_guidance else 1)
            accelerator.wait_for_everyone()

            for k in range(0, world_size):
                for context in global_context[k::world_size]:
                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :,
                                                            c] + noise_pred_gathered[k][:, :, c]
                        counter[:, :, c] = counter[:, :, c] + 1

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond_0, noise_pred_uncond_1, noise_pred_text = (
                    noise_pred / counter).chunk(3)
                noise_pred = noise_pred_uncond_0 + \
                    guidance_scale[0] * (noise_pred_uncond_1 - noise_pred_uncond_0) + \
                    guidance_scale[1] * (noise_pred_text - noise_pred_uncond_1)
                # noise_pred = noise_pred_text
            else:
                noise_pred /= counter
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample
            latents = self.merge_video_format(latents, 
                                              pose_image.shape[2] // self.vae_scale_factor,
                                              pose_image.shape[3] // self.vae_scale_factor,
                                              height, width, view_list)
            accelerator.wait_for_everyone()

        # Post-processing
        video = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        video = (video / 2 + 0.5).clamp(0, 1)
        accelerator.wait_for_everyone()
        return AnimationPipelineOutput(videos=video)

    def train(
            self,
            prompt: Union[str, List[str]],
            prompt_embeddings: Optional[torch.FloatTensor] = None,
            video_length: Optional[int] = 8,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            timestep: Union[torch.Tensor, float, int] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            controlnet_condition: list = None,
            init_latents: Optional[torch.FloatTensor] = None,
            appearance_encoder=None,
            source_image: torch.FloatTensor = None,
            clip_image: torch.FloatTensor = None,
            **kwargs,
    ):

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        prompt_embeds = self.negative_prompt_embeds
        prompt_attention_mask = self.negative_prompt_attention_mask
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        assert num_videos_per_prompt == 1
        control = controlnet_condition
        latents = init_latents
        del init_latents
        pose_guide_conditions = rearrange(control, "b f c h w -> b c f h w")

        prompt_embeds = self.image_encoder(clip_image, 
                                           return_dict=True).last_hidden_state
        t = timestep
        batch_size = pose_guide_conditions.shape[0]
        dropout_prompt_rate = 0.2
        dropout_pose_rate = 0.1 # or 0.05 for CFG 
        uncond_rate_prompt = torch.tensor(np.random.uniform(0, 2., batch_size) / 2. < dropout_prompt_rate)
        uncond_rate_pose = torch.tensor(np.random.uniform(0, 1., batch_size) / 1. < dropout_pose_rate)
        prompt_embeds[uncond_rate_prompt] = 0.
        pose_guide_conditions[uncond_rate_pose] = 0.
        
        # predict the noise residual
        noise_pred = self.unet(
            latents, 
            timestep=t, 
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            pose_guide_conditions=pose_guide_conditions,
        )[0]

        return noise_pred