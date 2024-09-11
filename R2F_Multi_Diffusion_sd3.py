# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import re
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import warnings
from dataclasses import dataclass

import PIL
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    AutoModelForZeroShotObjectDetection,
    AutoModelForMaskGeneration,
    AutoProcessor,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    SD3LoraLoaderMixin,
)
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    BaseOutput
)
from diffusers.utils.torch_utils import randn_tensor

from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
"""

@dataclass
class R2FMultiDiffusion3PipelineOutput(BaseOutput):
    images: List[PIL.Image.Image]
    bbox_images: List[PIL.Image.Image]
    object_images: List[PIL.Image.Image]
    masked_object_images: List[PIL.Image.Image]
    bbox_object_images: List[PIL.Image.Image]

def is_consecutive_words(s: str, t: str):
    words_s = re.findall(r'\b\w+\b', s)
    words_t = re.findall(r'\b\w+\b', t)
    len_s = len(words_s)
    len_t = len(words_t)

    if len_s > len_t:
        return False

    for i in range(len_t - len_s + 1):
        if words_t[i:i + len_s] == words_s:
            return True

    return False

@dataclass
class R2FMultiDiffusionObject:
    def __init__(
        self,
        prompt: str,
        rare: str,
        freq: str,
        visual_detail_level: int,
        bbox: List[float],
        rare_base: Optional[str] = None,
        freq_base: Optional[str] = None
    ):
        self.prompt = prompt
        if not isinstance(self.prompt, str):
            raise ValueError(f"Objects should have 'prompt' attribute of type 'str'")

        self.rare = rare
        if not isinstance(self.rare, str):
            raise ValueError(f"Objects should have 'rare' attribute of type 'str'")
        if not is_consecutive_words(self.rare, self.prompt):
            raise ValueError(f"'rare' should match some consecutive words of 'prompt',\
                but '{self.rare}' does not match some consecutive words of '{self.prompt}'")

        self.freq = freq
        if not isinstance(self.freq, str):
            raise ValueError(f"Objects should have 'freq' attribute of type 'str'")

        self.visual_detail_level = visual_detail_level
        if not isinstance(self.visual_detail_level, int):
            raise ValueError(f"Objects should have 'visual_detail_level' attribute of type 'int'")
        if self.visual_detail_level < 0 or self.visual_detail_level > 5:
            raise ValueError(f"'visual_detail_level' should be an integer between 0 and 5")

        self.bbox = bbox
        if not isinstance(self.bbox, list):
            raise ValueError(f"Objects should have 'bbox' attribute of type 'list'")
        if not all(isinstance(x, (int, float)) for x in self.bbox) or len(self.bbox) != 4:
            raise ValueError("'bbox' should be a list of four numbers")
        x_min, y_min, x_max, y_max = self.bbox
        if 0 > x_min or x_min > x_max or x_max > 1 or 0 > y_min or y_min > y_max or y_max > 1:
            raise ValueError(f"Invalid bbox ({bbox})")

        self.rare_base = rare_base
        self.freq_base = freq_base
        if self.rare_base:
            if not isinstance(self.rare_base, str):
                raise ValueError(f"'rare_base' attributes should be 'str' type")
            if not isinstance(self.freq_base, str):
                raise ValueError(f"Objects should have 'freq_base' attribute of type 'str' if they have 'rare_base'")

    @staticmethod
    def from_json(json_object: dict, validate: bool = True):
        if not isinstance(json_object, dict):
            raise ValueError(f"The given object should be of type 'dict'")

        return R2FMultiDiffusionObject(
            prompt=json_object.get("prompt"),
            rare=json_object.get("rare"),
            freq=json_object.get("freq"),
            rare_base=json_object.get("rare_base"),
            freq_base=json_object.get("freq_base"),
            visual_detail_level=json_object.get("visual_detail_level"),
            bbox= json_object.get("bbox")
        )

@dataclass
class R2FMultiDiffusionPrompt:
    def __init__(
        self,
        base_prompt: str,
        objects: List[R2FMultiDiffusionObject]
    ):
        self.base_prompt = base_prompt
        if not isinstance(self.base_prompt, str):
            raise ValueError(f"Base prompt should be of type 'str'")

        self.objects = objects
        if not isinstance(self.objects, list):
            raise ValueError(f"Objects should be of type 'list'")

        for obj in self.objects:
            if not isinstance(obj, R2FMultiDiffusionObject):
                raise ValueError(f"Each object should be of type 'R2FMultiDiffusionObject'")
            if obj.rare_base:
                if not is_consecutive_words(obj.rare_base, self.base_prompt):
                    raise ValueError(f"'rare_base' should match some consecutive words of the base prompt, \
                        but '{obj.rare_base}' does not match consecutive words of '{self.base_prompt}'")   
            else:
                if not is_consecutive_words(obj.rare, self.base_prompt):
                    raise ValueError(f"'rare' should match some consecutive words of the base prompt in case 'rare_base' is not provided, \
                        but '{obj.rare}' does not match consecutive words of '{self.base_prompt}'")        

    
    @staticmethod
    def from_json(json_object: dict):
        if not isinstance(json_object, dict):
            raise ValueError(f"The given object should be of type 'dict'")

        base_prompt = json_object.get("base_prompt")
        objects = json_object.get("objects")
        
        objects = [R2FMultiDiffusionObject.from_json(obj, validate=False) for obj in objects]

        return R2FMultiDiffusionPrompt(base_prompt=base_prompt, objects=objects)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class R2FMultiDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        detector_model: AutoModelForZeroShotObjectDetection,
        detector_processor: AutoProcessor,
        segmentor_model: AutoModelForMaskGeneration,
        segmentor_processor: AutoProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            detector_model=detector_model,
            detector_processor=detector_processor,
            segmentor_model=segmentor_model,
            segmentor_processor=segmentor_processor,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds
    
    def _get_t5_prompt_token_ids(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    1,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="do_not_pad",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        return text_input_ids

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds
    
    def _get_clip_prompt_token_ids(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        text_inputs = tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        return text_input_ids

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=0,
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            clip_model_index=1,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def check_inputs(
        self,
        height: int,
        width: int,
        num_inference_steps: int,
        visual_detail_level_to_transition_step: List[int],
        max_sequence_length: int = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
        for i in range(len(visual_detail_level_to_transition_step) - 1):
            if visual_detail_level_to_transition_step[i] >= visual_detail_level_to_transition_step[i + 1]:
                raise ValueError(f"`visual_detail_level_to_transition_step` should be an increasing sequence.")

        if visual_detail_level_to_transition_step[-1] >= num_inference_steps:
            raise ValueError(f"Elements of `visual_detail_level_to_transition_step` should be less than `num_inference_steps`.")


        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def _draw_object_bboxes(image, objects, centered=False):
        bbox_image = image.copy()
        draw = PIL.ImageDraw.Draw(bbox_image)
        width, height = bbox_image.size

        for obj in objects:
            x0, y0, x1, y1 = obj.bbox
            if centered:
                x0, x1 = (x0 - x1 + 1) / 2, (x1 - x0 + 1) / 2
                y0, y1 = (y0 - y1 + 1) / 2, (y1 - y0 + 1) / 2
            x0, x1 = round(x0 * width), round(x1 * width)
            y0, y1 = round(y0 * height), round(y1 * height)
            draw.rectangle([x0, y0, x1, y1], outline='red', width=5)
            draw.text([x0 + 5, y0 + 5], obj.rare, fill=(255, 0, 0))
        
        return bbox_image

    class CrossAttnSaver():
        def __init__(self, module: torch.nn.Module):
            self.module = module
        
        def hook_forward(self, attn, original_forward, name):

            def forward(
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **cross_attention_kwargs,
            ) -> torch.Tensor:
                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                context_input_ndim = encoder_hidden_states.ndim
                if context_input_ndim == 4:
                    batch_size, channel, height, width = encoder_hidden_states.shape
                    encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                    
                batch_size = hidden_states.shape[0]

                # `sample` projections.
                query = attn.to_q(hidden_states)

                # `context` projections.
                key = attn.add_k_proj(encoder_hidden_states)

                head_dim = query.shape[-1] // attn.heads
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Save cross attention weights
                # From torch.nn.functional.scaled_dot_product_attention
                scale_factor = 1 / math.sqrt(query.size(-1))
                cross_attn_weights = query @ key.transpose(-2, -1) * scale_factor
                cross_attn_weights = torch.softmax(cross_attn_weights, dim=-1) # ( batch, num_attn_heads, height * width, encoder_dim )

                self.saved_cross_attns[name] = cross_attn_weights

                return original_forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)

            return forward

        def __enter__(self):
            self.original_forward = {}
            self.saved_cross_attns = {}

            for name, submodule in self.module.named_modules():
                if "attn" in name and isinstance(submodule, Attention):
                    self.original_forward[name] = submodule.forward
                    submodule.forward = self.hook_forward(submodule, self.original_forward[name], name)
            
            return self
        
        def __exit__(self, exec_type, exec_val, traceback):
            for name, submodule in self.module.named_modules():
                if "attn" in name and isinstance(submodule, Attention):
                    submodule.forward = self.original_forward[name]
    
    def get_current_prompt_and_object_indices(
        self,
        prompt: str,
        objects: List[R2FMultiDiffusionObject],
        current_step: int,
        visual_detail_level_to_transition_step: List[int],
        alt_step: int,
        is_overall: bool,
    ):
        """
        For multi-object
        """
        ################################################################################
        # 1. Get current prompt                                                        #
        ################################################################################

        object_prompt_list = []
        for obj in objects:
            rare, freq = obj.rare, obj.freq
            if is_overall:
                rare = obj.rare_base or rare
                freq = obj.freq_base or freq

            transition_step = visual_detail_level_to_transition_step[obj.visual_detail_level]

            if current_step < transition_step and current_step % alt_step == 0:
                object_prompt = freq
                prompt = prompt.replace(rare, freq)
            else:
                object_prompt = rare

            object_prompt_list.append(object_prompt)
        
        ################################################################################
        # 1. Get object token indices                                                  #
        ################################################################################
        def first_occurence(A: torch.Tensor, B: torch.Tensor):
            return next((i for i in range(len(A) - len(B) + 1) if torch.equal(A[i:i+len(B)], B)), -1)

        prompt_clip_token_ids = self._get_clip_prompt_token_ids(prompt).squeeze()
        prompt_t5_token_ids = self._get_t5_prompt_token_ids(prompt).squeeze()

        object_indices_list = []
        for object_prompt in object_prompt_list:
            object_clip_token_ids = self._get_clip_prompt_token_ids(object_prompt).squeeze()
            object_clip_token_ids = object_clip_token_ids[1:-1] # remove start / end tokens for CLIPTokenizer

            pos = first_occurence(prompt_clip_token_ids, object_clip_token_ids)
            if pos < 0:
                raise ValueError(f"The object ({object_prompt}) must be a subsequence of the prompt ({prompt}).")
            
            clip_indices = torch.arange(pos, pos + len(object_clip_token_ids))

            object_t5_token_ids = self._get_t5_prompt_token_ids(object_prompt).squeeze()
            object_t5_token_ids = object_t5_token_ids[:-1] # remove end token for T5Tokenizer

            pos = first_occurence(prompt_t5_token_ids, object_t5_token_ids)
            if pos < 0:
                raise ValueError(f"The object ({object_prompt}) must be a subsequence of the prompt ({prompt}).")
            
            pos += self.tokenizer_max_length # add offset
            t5_indices = torch.arange(pos, pos + len(object_t5_token_ids))

            object_indices_list.append(torch.cat([clip_indices, t5_indices]))
        
        return prompt, object_indices_list
    
    @torch.enable_grad()   
    def guide_latents_with_bboxes(
        self,
        latent: torch.Tensor,
        timestep_args: Tuple[int, torch.Tensor, torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        object_indices_list: torch.Tensor,
        object_bbox_list: List[float],
        bbox_guidance_iterations: int,
        bbox_loss_scale: float,
    ):
        device = self._execution_device
        i, t, timestep = timestep_args

        def bbox_loss(
            cross_attn: torch.Tensor,       # (num_attn_blocks, num_attn_heads, height * width, num_indices)
            bbox: torch.Tensor
        ):
            height = width = round(cross_attn.shape[-2] ** 0.5)

            cross_attn = cross_attn.mean(dim=-1).mean(dim=0).mean(dim=0)

            x_min, y_min, x_max, y_max = bbox
            r_min, c_min, r_max, c_max = \
                round(y_min * height), round(x_min * width), \
                round(y_max * height), round(x_max * width)

            mask = torch.zeros((height, width)).to(device)
            mask[r_min:r_max, c_min:c_max] = 1
            mask = mask.view(-1)

            # LayoutControl loss (arxiv.org/abs/2304.03373)
            bbox_loss = ((((1 - mask) * cross_attn).sum(dim=-1) / (1 + cross_attn.sum(dim=-1))) ** 2).mean()

            return bbox_loss * bbox_loss_scale

        for it in range(bbox_guidance_iterations):
            latent.requires_grad_(True)

            with self.CrossAttnSaver(self.transformer) as cross_attn_saver:
                self.transformer(
                    hidden_states=latent,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False
                )
                
                object_cross_attn = torch.stack(
                    [cross_attn[1] for cross_attn in cross_attn_saver.saved_cross_attns.values()],
                    dim=0
                ) # (num_attn_blocks, num_attn_heads, height * width, encoder_dim)

            loss = 0
            for object_indices, bbox in zip(object_indices_list, object_bbox_list):
                loss += bbox_loss(object_cross_attn[:, :, :, object_indices], bbox)

            loss /= len(object_indices_list)

            loss.backward()
            latent_grad = latent.grad

            latent.requires_grad_(False)

            self.transformer.zero_grad(set_to_none=True)

            if hasattr(self.scheduler, 'sigmas'):
                latent = latent - latent_grad * self.scheduler.sigmas[i] ** 2
            elif hasattr(self.scheduler, 'alphas_cumprod'):
                warnings.warn("Using guidance scaled with alphas_cumprod")
                # Scaling with classifier guidance
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                # Classifier guidance: https://arxiv.org/pdf/2105.05233.pdf
                # DDIM: https://arxiv.org/pdf/2010.02502.pdf
                scale = (1 - alpha_prod_t) ** (0.5)
                latent = latent - scale * latent_grad
            else:
                # NOTE: no scaling is performed
                warnings.warn("No scaling in guidance is performed")
                latent = latent - latent_grad
        
        return latent

    def generate_single_object(
        self,
        r2f_multi_object: R2FMultiDiffusionObject,
        bbox: List[float],
        negative_prompt: str,
        height: int,
        width: int,
        timesteps: List[int],
        num_inference_steps: int,
        visual_detail_level_to_transition_step: List[int],
        alt_step: int,
        bbox_guidance_steps: int,
        bbox_guidance_iterations: int,
        bbox_loss_scale: float,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        max_sequence_length: int = 256,
        output_type: Optional[str] = "pil",
        use_centered_bbox: bool = True,
    ):
        """
        Generate single object latents and image
        """
        device = self._execution_device
        
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        # Prepare negative prompt embeddings
        if self.do_classifier_free_guidance:
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                device=device,
                clip_skip=self.clip_skip,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        # Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
        )

        latents_by_step = []
        cross_attn_by_step = []
        
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                object_prompt, object_indices_list = self.get_current_prompt_and_object_indices(
                    prompt=r2f_multi_object.prompt,
                    objects=[r2f_multi_object],
                    current_step=i,
                    visual_detail_level_to_transition_step=visual_detail_level_to_transition_step,
                    alt_step=alt_step,
                    is_overall=False
                )

                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                    prompt=object_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    device=device,
                    clip_skip=self.clip_skip,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # bbox guidance
                if i < bbox_guidance_steps:
                    latents = self.guide_latents_with_bboxes(
                        latent=latents,
                        timestep_args=(i, t, timestep),
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        object_indices_list=object_indices_list,
                        object_bbox_list=[bbox],
                        bbox_guidance_iterations=bbox_guidance_iterations,
                        bbox_loss_scale=bbox_loss_scale,
                    )
                
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                latents_by_step.append(latents.detach().cpu())

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        scaled_latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(scaled_latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        image = image[0]

        return image, latents_by_step
        
    def get_single_object_mask(
        self,
        image: PIL.Image,
        object_prompt: str,
    ):
        device = self._execution_device

        detector_inputs = self.detector_processor(
            images=image, 
            text=object_prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            detector_outputs = self.detector_model(**detector_inputs)
        
        detector_results = self.detector_processor.post_process_grounded_object_detection(
            detector_outputs,
            detector_inputs.input_ids,
            box_threshold=0,
            text_threshold=0,
            target_sizes=[image.size[::-1]]
        )[0]
        idx = detector_results["scores"].argmax().item()
        object_box = detector_results["boxes"][idx].tolist()

        segmentor_inputs = self.segmentor_processor(
            images=image,
            input_boxes=[[object_box]],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            segmentor_outputs = self.segmentor_model(**segmentor_inputs)

        masks = self.segmentor_processor.image_processor.post_process_masks(
            segmentor_outputs.pred_masks.cpu(),
            segmentor_inputs.original_sizes.cpu(),
            segmentor_inputs.reshaped_input_sizes.cpu()
        )[0]
        object_mask = masks[0][0]

        return object_mask
    
    def refine_latents_and_mask(
        self,
        latents_by_step: List[torch.Tensor],
        object_mask: torch.Tensor,
        shifts: Optional[List[float]] = None,
    ):
        # Reshape mask
        latent_height, latent_width = latents_by_step[-1].shape[-2:]
        object_mask = object_mask.float().unsqueeze(0).unsqueeze(0) # (1, 1, height, width)
        object_mask = torch.nn.functional.interpolate(
            object_mask,
            size=(latent_height, latent_width),
            mode='bilinear',
            align_corners=False
        ) # (1, 1, latent_height, latent_width)
        object_mask = object_mask.to(dtype=self.text_encoder.dtype)

        if shifts:
            latent_shifts = (round(shifts[0] * latent_height), round(shifts[1] * latent_width))
            latents_by_step = [
                torch.roll(latents, shifts=latent_shifts, dims=(-2, -1)) for latents in latents_by_step
            ]

            padded_mask = torch.cat([object_mask, torch.zeros(object_mask.shape)], dim=-2)
            padded_mask = torch.cat([padded_mask, torch.zeros(padded_mask.shape)], dim=-1)
            padded_mask = torch.roll(padded_mask, shifts=latent_shifts, dims=(-2, -1))
            object_mask = padded_mask[:, :, :object_mask.shape[-2], :object_mask.shape[-1]]

        return latents_by_step, object_mask
    
    def generate_overall(
        self,
        r2f_multi_prompt: R2FMultiDiffusionPrompt,
        object_latents_list_by_step: List[List[torch.Tensor]],
        object_bbox_list: List[List[float]],
        object_mask_list: List[torch.Tensor],
        negative_prompt: str,
        height: int,
        width: int,
        timesteps: List[int],
        num_inference_steps: int,
        visual_detail_level_to_transition_step: List[int],
        alt_step: int,
        latent_fusion_steps: int,
        bbox_guidance_steps: int,
        bbox_guidance_iterations: int,
        bbox_loss_scale: float,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_sequence_length: int = 256,
        output_type: Optional[str] = "pil",
    ):
        """
        Generate overall image
        """
        device = self._execution_device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        object_mask_list = [ object_mask.to(device) for object_mask in object_mask_list ]

        # Prepare negative prompt embeddings
        if self.do_classifier_free_guidance:
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                device=device,
                clip_skip=self.clip_skip,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        # Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
        )

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                prompt, object_indices_list = self.get_current_prompt_and_object_indices(
                    prompt=r2f_multi_prompt.base_prompt,
                    objects=r2f_multi_prompt.objects,
                    current_step=i,
                    visual_detail_level_to_transition_step=visual_detail_level_to_transition_step,
                    alt_step=alt_step,
                    is_overall=True
                )

                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                    prompt=prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    device=device,
                    clip_skip=self.clip_skip,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                if i < bbox_guidance_steps:
                    # Bbox guidance
                    latents = self.guide_latents_with_bboxes(
                        latent=latents,
                        timestep_args=(i, t, timestep),
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        object_indices_list=object_indices_list,
                        object_bbox_list=object_bbox_list,
                        bbox_guidance_iterations=bbox_guidance_iterations,
                        bbox_loss_scale=bbox_loss_scale,
                    )

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]                    

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
                
                if i < latent_fusion_steps:
                    # Fuse latents
                    object_latents_list = [ latents.to(device) for latents in object_latents_list_by_step[i] ]
                    mask_sum = sum(object_mask_list)
                    latents -= mask_sum * latents
                    latents += sum(
                        lt * mask for lt, mask in zip(object_latents_list, object_mask_list)
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        image = image[0]
        
        return image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        r2f_multi_prompt: R2FMultiDiffusionPrompt,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        visual_detail_level_to_transition_step: List[int] = [0, 5, 10, 20, 30, 40],
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[str] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        alt_step: int = 2,
        max_sequence_length: int = 256,
        latent_fusion_steps: int = 15,
        bbox_guidance_steps: int = 10,
        bbox_guidance_iterations: int = 5,
        bbox_loss_scale: float = 30,
        use_centered_bbox: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Examples:
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        ################################################################################
        # 1. Check inputs. Raise error if not correct                                  #
        ################################################################################
        if not isinstance(r2f_multi_prompt, R2FMultiDiffusionPrompt):
            raise ValueError("r2f_multi_prompt should be of type 'R2FMultiDiffusionPrompt")
        self.check_inputs(
            height,
            width,
            num_inference_steps=num_inference_steps,
            visual_detail_level_to_transition_step=visual_detail_level_to_transition_step,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if self.do_classifier_free_guidance and not negative_prompt:
            negative_prompt = ""

        images = []
        bbox_images = []
        object_images = []
        masked_object_images = []
        bbox_object_images = []

        object_mask_list = []
        object_bbox_list = []
        object_latents_list_by_step = [[] for _ in range(num_inference_steps)]

        for r2f_multi_object in r2f_multi_prompt.objects:
            bbox = r2f_multi_object.bbox

            if use_centered_bbox:
                bbox_input = [
                    (1 - bbox[2] + bbox[0]) / 2,
                    (1 - bbox[3] + bbox[1]) / 2,
                    (1 + bbox[2] - bbox[0]) / 2,
                    (1 + bbox[3] - bbox[1]) / 2
                ]
            else:
                bbox_input = bbox

            image, latents_by_step = self.generate_single_object(
                r2f_multi_object=r2f_multi_object,
                bbox=bbox_input,
                height=height,
                width=width,
                timesteps=timesteps,
                generator=generator,
                num_inference_steps=num_inference_steps,
                visual_detail_level_to_transition_step=visual_detail_level_to_transition_step,
                negative_prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                output_type=output_type,
                alt_step=alt_step,
                use_centered_bbox=use_centered_bbox,
                bbox_guidance_steps=bbox_guidance_steps,
                bbox_guidance_iterations=bbox_guidance_iterations,
                bbox_loss_scale=bbox_loss_scale,
            )

            object_mask = self.get_single_object_mask(image, r2f_multi_object.rare)
            masked_image = np.array(image)
            masked_image[~object_mask] = 0
            masked_image = PIL.Image.fromarray(masked_image)

            shifts = None
            if use_centered_bbox:
                shifts = ((bbox[1] + bbox[3] - 1) / 2, (bbox[0] + bbox[2] - 1) / 2)

            latents_by_step, object_mask = self.refine_latents_and_mask(
                latents_by_step=latents_by_step,
                object_mask=object_mask,
                shifts=shifts
            )

            object_mask_list.append(object_mask)
            object_bbox_list.append(bbox)
            for latents_list, latents in zip(object_latents_list_by_step, latents_by_step):
                latents_list.append(latents)
                
            object_images.append(image)
            masked_object_images.append(masked_image)
            bbox_object_images.append(self._draw_object_bboxes(image, [r2f_multi_object], centered=True))
            
        mask_sum = sum(object_mask_list)
        for object_mask in object_mask_list:
            object_mask = torch.where(mask_sum != 0, object_mask / mask_sum, 0)

        image = self.generate_overall(
            height=height,
            width=width,
            r2f_multi_prompt=r2f_multi_prompt,
            object_latents_list_by_step=object_latents_list_by_step,
            object_bbox_list=object_bbox_list,
            object_mask_list=object_mask_list,
            timesteps=timesteps,
            generator=generator,
            num_inference_steps=num_inference_steps,
            visual_detail_level_to_transition_step=visual_detail_level_to_transition_step,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            output_type=output_type,
            alt_step=alt_step,
            latent_fusion_steps=latent_fusion_steps,
            bbox_guidance_steps=bbox_guidance_steps,
            bbox_guidance_iterations=bbox_guidance_iterations,
            bbox_loss_scale=bbox_loss_scale,
        )

        # TODO: support multiple images
        images = [image]
        bbox_images = [self._draw_object_bboxes(image, r2f_multi_prompt.objects)]
        object_images = [object_images]
        masked_object_images = [masked_object_images]
        bbox_object_images = [bbox_object_images]

        # Offload all models
        self.maybe_free_model_hooks()

        return R2FMultiDiffusion3PipelineOutput(
            images=images,
            bbox_images=bbox_images,
            object_images=object_images,
            masked_object_images=masked_object_images,
            bbox_object_images=bbox_object_images,
        )