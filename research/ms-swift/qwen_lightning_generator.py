import torch
import math
import os
from PIL import Image
import numpy as np
from typing import List, Optional, Dict, Any
import threading

class QwenImageEditGenerator:
    """
    Local image editor using Qwen-Image-Edit (normal QwenImageEditPipeline, NOT Lightning/Plus).
    Supports single-image editing with reference image.
    """

    def __init__(self, model_path: str, gpu_id: int=4, gpu_ids: Optional[List[int]]=None):
        self.model_path = model_path
        self.gpu_ids = gpu_ids if gpu_ids else [gpu_id]
        self.dtype = torch.bfloat16
        self.device = f'cuda:{self.gpu_ids[0]}'
        self._lock = threading.RLock()
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        from diffusers import DiffusionPipeline
        if len(self.gpu_ids) > 1:
            mem_per_gpu = float(os.environ.get('QWEN_EDIT_MAX_MEM_GIB', '80'))
            max_memory = {gid: f'{int(mem_per_gpu)}GiB' for gid in self.gpu_ids}
            print(f"[Qwen-Image-Edit] Loading with device_map='balanced' across GPUs {self.gpu_ids}")
            print(f'[Qwen-Image-Edit] max_memory = {max_memory}')
            pipeline = DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map='balanced', max_memory=max_memory, trust_remote_code=True)
        else:
            device = f'cuda:{self.gpu_ids[0]}'
            print(f'[Qwen-Image-Edit] Loading on single GPU {self.gpu_ids[0]}...')
            pipeline = DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True)
            pipeline.to(device)
        self.pipeline = pipeline
        print(f'[Qwen-Image-Edit] Pipeline class: {type(pipeline).__name__}')
        print(f'[Qwen-Image-Edit] Loaded successfully on GPUs {self.gpu_ids}!')

    def generate(self, prompt: str, images: Optional[List[Any]]=None, model: str='qwen-edit', cid: Optional[str]=None, image_format: str='url', output_dir: str='/tmp/qwen_agent_outputs') -> Dict[str, Any]:
        """Single-image editing. Takes the first input image as reference."""
        target_width, target_height = (1024, 1024)
        input_image = None
        if images and len(images) > 0:
            first = images[0]
            try:
                if isinstance(first, str):
                    if first.startswith(('http://', 'https://')):
                        import requests
                        from io import BytesIO
                        resp = requests.get(first)
                        input_image = Image.open(BytesIO(resp.content)).convert('RGB')
                    else:
                        input_image = Image.open(first).convert('RGB')
                elif isinstance(first, Image.Image):
                    input_image = first.convert('RGB')
            except Exception as e:
                print(f'[Qwen Edit] Error loading image: {e}')
        if input_image is None:
            print('[Qwen Edit] No input image. Using dummy black image.')
            input_image = Image.fromarray(np.zeros((target_height, target_width, 3), dtype=np.uint8))
        num_inference_steps = int(os.environ.get('QWEN_IMAGE_STEPS', '50'))
        cfg_scale = float(os.environ.get('QWEN_IMAGE_CFG', '4.0'))
        negative_prompt = ' '
        if self.pipeline is None:
            raise ValueError('Qwen-Image-Edit pipeline not initialized.')
        generator = torch.Generator(device=self.device).manual_seed(42)
        print(f'[Qwen Edit] steps={num_inference_steps}, cfg={cfg_scale}, image_size={input_image.size}')
        print(f'[Qwen Edit] prompt: {prompt[:150]}...')
        with self._lock:
            with torch.inference_mode():
                output = self.pipeline(prompt=prompt, image=input_image, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, true_cfg_scale=cfg_scale, generator=generator, num_images_per_prompt=1)
                output_images = output.images
        os.makedirs(output_dir, exist_ok=True)
        filename = f"qwen_edit_{cid or 'gen'}_{int(torch.randint(0, 1000000, (1,)).item())}.png"
        filepath = os.path.join(output_dir, filename)
        output_images[0].save(filepath)
        return {'code': 0, 'msg': 'success', 'data': {'urls': [f'file://{filepath}'], 'local_path': filepath}}

class QwenImageGenerator:
    """
    Local text-to-image generator using Qwen-Image (QwenImagePipeline).
    Pure generation — no input images required.
    """

    def __init__(self, model_path: str, gpu_id: int=3, gpu_ids: Optional[List[int]]=None):
        self.model_path = model_path
        self.gpu_ids = gpu_ids if gpu_ids else [gpu_id]
        self.dtype = torch.bfloat16
        self.device = f'cuda:{self.gpu_ids[0]}'
        self._lock = threading.RLock()
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        from diffusers import DiffusionPipeline
        if len(self.gpu_ids) > 1:
            mem_per_gpu = float(os.environ.get('QWEN_GEN_MAX_MEM_GIB', '80'))
            max_memory = {gid: f'{int(mem_per_gpu)}GiB' for gid in self.gpu_ids}
            print(f"[Qwen-Image] Loading with device_map='balanced' across GPUs {self.gpu_ids}")
            print(f'[Qwen-Image] max_memory = {max_memory}')
            pipeline = DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map='balanced', max_memory=max_memory, trust_remote_code=True)
        else:
            device = f'cuda:{self.gpu_ids[0]}'
            print(f'[Qwen-Image] Loading on single GPU {self.gpu_ids[0]}...')
            pipeline = DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True)
            pipeline.to(device)
        self.pipeline = pipeline
        print(f'[Qwen-Image] Pipeline class: {type(pipeline).__name__}')
        print(f'[Qwen-Image] Loaded successfully on GPUs {self.gpu_ids}!')

    def generate(self, prompt: str, images: Optional[List[Any]]=None, model: str='qwen-gen', cid: Optional[str]=None, image_format: str='url', output_dir: str='/tmp/qwen_agent_outputs') -> Dict[str, Any]:
        """Text-to-image generation. The `images` parameter is ignored."""
        target_width, target_height = (1024, 1024)
        num_inference_steps = int(os.environ.get('QWEN_IMAGE_STEPS', '50'))
        cfg_scale = float(os.environ.get('QWEN_IMAGE_CFG', '4.0'))
        negative_prompt = ' '
        if self.pipeline is None:
            raise ValueError('Qwen-Image pipeline not initialized.')
        generator = torch.Generator(device=self.device).manual_seed(42)
        clean_prompt = prompt.replace('<image>', '').replace('<image>\n', '').strip()
        print(f'[Qwen Gen] steps={num_inference_steps}, cfg={cfg_scale}')
        print(f'[Qwen Gen] prompt: {clean_prompt[:150]}...')
        with self._lock:
            with torch.inference_mode():
                output = self.pipeline(prompt=clean_prompt, negative_prompt=negative_prompt, height=target_height, width=target_width, num_inference_steps=num_inference_steps, true_cfg_scale=cfg_scale, generator=generator, num_images_per_prompt=1)
                output_images = output.images
        os.makedirs(output_dir, exist_ok=True)
        filename = f"qwen_gen_{cid or 'gen'}_{int(torch.randint(0, 1000000, (1,)).item())}.png"
        filepath = os.path.join(output_dir, filename)
        output_images[0].save(filepath)
        return {'code': 0, 'msg': 'success', 'data': {'urls': [f'file://{filepath}'], 'local_path': filepath}}
QwenLightningImageGenerator = QwenImageEditGenerator
