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
class R2FplusDiffusion3PipelineOutput(BaseOutput):
    images: List[PIL.Image.Image]
    bbox_images: List[PIL.Image.Image]
    object_images: List[Dict[str, PIL.Image.Image]]
    masked_object_images: List[Dict[str, PIL.Image.Image]]
    bbox_object_images: List[Dict[str, PIL.Image.Image]]


@dataclass
class R2FplusDiffusionObject:
    def __init__(
        self,
        prompt: str,
        object: str,
        r2f: List[str],
        visual_detail_levels: List[int],
        bbox: List[float],
        object_key: Optional[str] = None
    ):
        self.prompt = prompt
        if not isinstance(prompt, str):
            raise ValueError(f"Objects should have 'prompt' attribute of type 'str'")

        keys = re.findall(r'#\d+', prompt)
        if len(keys) != 1:
            raise ValueError(f"The object prompt should have exactly one key of form '#N', where N is a number")
        if object_key and keys[0] != object_key:
            raise ValueError(f"The key in the object prompt ({keys[0]}) does not match the object key ({object_key})")

        self.object = object
        if not isinstance(object, str):
            raise ValueError(f"Objects should have 'object' attribute of type 'str'")
        
        self.r2f = r2f
        if not isinstance(r2f, list):
            raise ValueError(f"Objects should have 'r2f' attribute of type 'list'")
        for r2f_prompt in r2f:
            if not isinstance(r2f_prompt, str):
                raise ValueError(f"Each object r2f prompt should be of type 'str'")

        self.visual_detail_levels = visual_detail_levels
        if not isinstance(visual_detail_levels, list):
            raise ValueError(f"Objects should have 'visual_detail_levels' attribute of type 'list'")
        if len(r2f) != len(visual_detail_levels):
            raise ValueError(f"The number of visual detail levels should be equal to the number of r2f prompts.")
        for visual_detail_level in visual_detail_levels:
            if not isinstance(visual_detail_level, int) or visual_detail_level < 0 or visual_detail_level > 5:
                raise ValueError(f"Each visual_detail_level should be an integer between 0 and 5.")
        if visual_detail_levels != sorted(visual_detail_levels):
                raise ValueError(f"Visual detail levels should be increasing.")
            
        self.bbox = bbox
        if not isinstance(self.bbox, list):
            raise ValueError(f"Objects should have 'bbox' attribute of type 'list'")
        if not all(isinstance(x, (int, float)) for x in self.bbox) or len(self.bbox) != 4:
            raise ValueError("'bbox' should be a list of four numbers")
        x_center, y_center, bbox_width, bbox_height = self.bbox
        x_min, x_max = x_center - bbox_width / 2, x_center + bbox_width / 2
        y_min, y_max = y_center - bbox_height / 2, y_center + bbox_height / 2
        if 0 > x_min or x_min > x_max or x_max > 1 or 0 > y_min or y_min > y_max or y_max > 1:
            raise ValueError(f"Invalid bbox ({bbox})")

    @staticmethod
    def from_json(json_object: dict, object_key: Optional[str] = None):
        if not isinstance(json_object, dict):
            raise ValueError(f"The given object should be of type 'dict'")

        return R2FplusDiffusionObject(
            prompt=json_object.get("prompt"),
            object=json_object.get("object"),
            r2f=json_object.get("r2f"),
            visual_detail_levels=json_object.get("visual_detail_levels"),
            bbox=json_object.get("bbox"),
            object_key=object_key
        )

@dataclass
class R2FplusDiffusionPrompt:
    def __init__(
        self,
        original_prompt: str,
        base_prompt: str,
        objects: Dict[str, R2FplusDiffusionObject]
    ):
        self.original_prompt = original_prompt
        if not isinstance(self.original_prompt, str):
            raise ValueError(f"The original prompt should be of type 'str'")

        self.base_prompt = base_prompt
        if not isinstance(self.base_prompt, str):
            raise ValueError(f"Base prompt should be of type 'str'")

        self.objects = objects
        if not isinstance(self.objects, dict):
            raise ValueError(f"Objects should be of type 'dict'")
        for obj in objects.values():
            if not isinstance(obj, R2FplusDiffusionObject):
                raise ValueError(f"Each object should be of type 'R2FPlusDiffusionObject'")
        
        keys = re.findall(r'#\d+', base_prompt)
        if len(keys) != len(objects):
            raise ValueError(f"The number of keys in the base_prompt does not match the number of objects.")
        for i in range(len(objects)):
            object_key = f"#{i + 1}"
            if not object_key in keys:
                raise ValueError(f"The base_prompt should include keys for all objects, formatted as #1, #2, and so on, but the key '{object_key}' is missing.")
            if not object_key in objects:
                raise ValueError(f"The object of key '{object_key}' is missing.")


    @staticmethod
    def from_json(json_object: dict):
        if not isinstance(json_object, dict):
            raise ValueError(f"The given object should be of type 'dict'")

        original_prompt = json_object.get("original_prompt")
        base_prompt = json_object.get("base_prompt")
        objects = json_object.get("objects")

        objects = {key: R2FplusDiffusionObject.from_json(obj, key) for key, obj in objects.items()}

        return R2FplusDiffusionPrompt(
            original_prompt=original_prompt,
            base_prompt=base_prompt,
            objects=objects
        )


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


class R2FplusDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
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

        for object_key, obj in objects.items():
            x_center, y_center, bbox_width, bbox_height = obj.bbox
            if centered:
                x_center, y_center = 0.5, 0.5
            x_min, x_max = x_center - bbox_width / 2, x_center + bbox_width / 2
            y_min, y_max = y_center - bbox_height / 2, y_center + bbox_height / 2
            x_min, x_max = round(x_min * width), round(x_max * width)
            y_min, y_max = round(y_min * height), round(y_max * height)
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=5)
            draw.text([x_min + 5, y_min + 5], f'{object_key}: {obj.object}', fill='red')
        
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
        objects: Dict[str, R2FplusDiffusionObject],
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
        object_keys = re.findall(r'#\d+', prompt)

        clip_token_ids = self._get_clip_prompt_token_ids(prompt).squeeze()
        t5_token_ids = self._get_t5_prompt_token_ids(prompt).squeeze()

        object_indices_dict = {}
        for object_key in object_keys:
            obj = objects[object_key]

            r2f_idx = 0
            while r2f_idx < len(obj.r2f) \
                and current_step >= visual_detail_level_to_transition_step[obj.visual_detail_levels[r2f_idx]]:
                r2f_idx += 1

            if r2f_idx < len(obj.r2f) and current_step % alt_step == 0:
                object_prompt = obj.r2f[r2f_idx]
            else:
                object_prompt = obj.object

            next_prompt = prompt.replace(object_key, object_prompt)
            next_clip_token_ids = self._get_clip_prompt_token_ids(next_prompt).squeeze()
            next_t5_token_ids = self._get_t5_prompt_token_ids(next_prompt).squeeze()

            # clip indices
            clip_start_idx = 0
            while clip_start_idx < len(next_clip_token_ids) \
                and clip_token_ids[clip_start_idx] == next_clip_token_ids[clip_start_idx]:
                clip_start_idx += 1
            
            clip_end_idx = 0
            while clip_end_idx > -len(next_clip_token_ids) \
                and clip_token_ids[clip_end_idx - 1] == next_clip_token_ids[clip_end_idx - 1]:
                clip_end_idx -= 1
            clip_end_idx += len(next_clip_token_ids)

            clip_indices = torch.arange(clip_start_idx, clip_end_idx)

            # t5 indices
            t5_start_idx = 0
            while t5_start_idx < len(next_t5_token_ids) \
                and t5_token_ids[t5_start_idx] == next_t5_token_ids[t5_start_idx]:
                t5_start_idx += 1
            
            t5_end_idx = 0
            while t5_end_idx > -len(next_t5_token_ids) \
                and t5_token_ids[t5_end_idx - 1] == next_t5_token_ids[t5_end_idx - 1]:
                t5_end_idx -= 1
            t5_end_idx += len(next_t5_token_ids)

            t5_indices = torch.arange(t5_start_idx, t5_end_idx) + self.tokenizer_max_length

            object_indices_dict[object_key] = torch.cat([clip_indices, t5_indices])

            prompt = next_prompt
            clip_token_ids = next_clip_token_ids
            t5_token_ids = next_t5_token_ids
    
        return prompt, object_indices_dict
    
    @torch.enable_grad()   
    def guide_latents_with_bboxes(
        self,
        latent: torch.Tensor,
        timestep_args: Tuple[int, torch.Tensor, torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        object_indices_dict: Dict[str, torch.Tensor],
        object_bbox_dict: Dict[str, List[float]],
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

            x_center, y_center, bbox_width, bbox_height = bbox
            x_min, x_max = x_center - bbox_width / 2, x_center + bbox_width / 2
            y_min, y_max = y_center - bbox_height / 2, y_center + bbox_height / 2
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
            for object_key in object_bbox_dict.keys():
                object_indices = object_indices_dict[object_key]
                bbox = object_bbox_dict[object_key]
                loss += bbox_loss(object_cross_attn[:, :, :, object_indices], bbox)

            loss /= len(object_indices_dict)

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
        object_key: str,
        r2fplus_object: R2FplusDiffusionObject,
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

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                object_prompt, object_indices_dict = self.get_current_prompt_and_object_indices(
                    prompt=r2fplus_object.prompt,
                    objects={object_key: r2fplus_object},
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
                        object_indices_dict=object_indices_dict,
                        object_bbox_dict={object_key: bbox},
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
        r2fplus_prompt: R2FplusDiffusionPrompt,
        object_latents_dict_by_step: List[Dict[str, torch.Tensor]],
        object_bbox_dict: Dict[str, List[float]],
        object_mask_dict: Dict[str, torch.Tensor],
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

        object_mask_dict = {key: object_mask.to(device) for key, object_mask in object_mask_dict.items()}

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

                prompt, object_indices_dict = self.get_current_prompt_and_object_indices(
                    prompt=r2fplus_prompt.base_prompt,
                    objects=r2fplus_prompt.objects,
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
                        object_indices_dict=object_indices_dict,
                        object_bbox_dict=object_bbox_dict,
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
                
                # Fuse latents
                if i < latent_fusion_steps:
                    object_latents_dict = object_latents_dict_by_step[i]
                    for object_key in object_mask_dict.keys():
                        object_latent = object_latents_dict[object_key].to(device)
                        mask = object_mask_dict[object_key]
                        latents *= 1 - mask
                        latents += object_latent * mask

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
        r2fplus_prompt: R2FplusDiffusionPrompt,
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
        latent_fusion_steps: int = 20,
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
        
        for param in self.transformer.parameters():
            param.requires_grad = False

        ################################################################################
        # 1. Check inputs. Raise error if not correct                                  #
        ################################################################################
        if not isinstance(r2fplus_prompt, R2FplusDiffusionPrompt):
            raise ValueError("r2fplus_prompt should be of type 'R2FPlusDiffusionPrompt")
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

        object_images = {}
        masked_object_images = {}
        bbox_object_images = {}

        object_mask_dict = {}
        object_latents_dict_by_step = [{} for _ in range(num_inference_steps)]

        if latent_fusion_steps > 0:
            for object_key, r2fplus_object in r2fplus_prompt.objects.items():
                bbox = r2fplus_object.bbox

                if use_centered_bbox:
                    bbox_input = [0.5, 0.5, bbox[2], bbox[3]]
                else:
                    bbox_input = bbox

                image, latents_by_step = self.generate_single_object(
                    object_key=object_key,
                    r2fplus_object=r2fplus_object,
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

                object_mask = self.get_single_object_mask(image, r2fplus_object.object)

                masked_image = np.array(image)
                masked_image[~object_mask] = 0
                masked_image = PIL.Image.fromarray(masked_image)

                bbox_image = self._draw_object_bboxes(
                    image,
                    {object_key: r2fplus_object},
                    centered=use_centered_bbox
                )

                shifts = None
                if use_centered_bbox:
                    shifts = (bbox[1] - 0.5, bbox[0] - 0.5)

                latents_by_step, object_mask = self.refine_latents_and_mask(
                    latents_by_step=latents_by_step,
                    object_mask=object_mask,
                    shifts=shifts
                )

                object_mask_dict[object_key] = object_mask
                for latents_dict, latents in zip(object_latents_dict_by_step, latents_by_step):
                    latents_dict[object_key] = latents
                    
                object_images[object_key] = image
                masked_object_images[object_key] = masked_image
                bbox_object_images[object_key] = bbox_image
        
        object_bbox_dict = {key: obj.bbox for key, obj in r2fplus_prompt.objects.items()}

        image = self.generate_overall(
            height=height,
            width=width,
            r2fplus_prompt=r2fplus_prompt,
            object_bbox_dict=object_bbox_dict,
            object_latents_dict_by_step=object_latents_dict_by_step,
            object_mask_dict=object_mask_dict,
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
        bbox_images = [self._draw_object_bboxes(image, r2fplus_prompt.objects)]
        object_images = [object_images]
        masked_object_images = [masked_object_images]
        bbox_object_images = [bbox_object_images]

        # Offload all models
        self.maybe_free_model_hooks()

        return R2FplusDiffusion3PipelineOutput(
            images=images,
            bbox_images=bbox_images,
            object_images=object_images,
            masked_object_images=masked_object_images,
            bbox_object_images=bbox_object_images,
        )