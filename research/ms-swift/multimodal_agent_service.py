"""
Multimodal dialogue agent service.
Main capabilities:
1. Session management: users can create new sessions and clear the memory bank.
2. Memory bank management: store summary information for each image.
3. VLM A invocation: summarize uploaded or generated images.
"""
import os
import json
import re
import uuid
import threading
import requests
import html
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
DEFAULT_GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
os.environ['CUDA_VISIBLE_DEVICES'] = DEFAULT_GPUS
os.environ['MAX_PIXELS'] = '100352000'
import torch
from swift.llm import PtEngine, RequestConfig, InferRequest, BaseArguments, get_template, get_model_tokenizer
from swift.tuners import Swift

def _make_thumbnail(image_path: str, size: int=448) -> str:
    """Create a square thumbnail cached under /tmp. Returns the thumbnail path."""
    import hashlib
    from PIL import Image as PILImage
    cache_dir = Path('/tmp/agent_thumbs_448')
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(image_path.encode()).hexdigest()[:12]
    thumb = cache_dir / f'thumb_{digest}_{size}.png'
    if thumb.exists():
        return str(thumb)
    try:
        img = PILImage.open(image_path).convert('RGB')
        if img.size != (size, size):
            img = img.resize((size, size), resample=PILImage.LANCZOS)
        img.save(str(thumb))
    except Exception as e:
        print(f'[Thumbnail] Failed to create thumbnail for {image_path}: {e}')
        return image_path
    return str(thumb)

class MemoryBank:
    """Memory bank manager."""

    def __init__(self):
        self.memories: List[Dict] = []
        self.next_id = 0

    def add_memory(self, memory: Dict) -> int:
        """
        Add a memory entry to the bank.
        
        Args:
            memory: Dict containing title, description, caption, and tags.
        
        Returns:
            The assigned memory ID.
        """
        memory['id'] = self.next_id
        memory['created_at'] = datetime.now().isoformat()
        self.memories.append(memory)
        self.next_id += 1
        return memory['id']

    def get_memory(self, memory_id: int) -> Optional[Dict]:
        """Get a memory entry by its ID."""
        for mem in self.memories:
            if mem.get('id') == memory_id:
                return mem
        return None

    def get_all_memories(self) -> List[Dict]:
        """Get a shallow copy of all memories."""
        return self.memories.copy()

    def clear(self):
        """Clear the memory bank."""
        self.memories = []
        self.next_id = 0

    def to_dict(self) -> Dict:
        """Convert the memory bank to a serializable dict."""
        return {'memories': self.memories, 'count': len(self.memories)}

class VLMModel:
    """Wrapper for a VLM model with optional multi-GPU load balancing."""

    def __init__(self, checkpoint_path: str, model_name: str='VLM', gpu_ids: Optional[List[int]]=None):
        """
        Initialize the VLM model.
        
        Args:
            checkpoint_path: Path to the model checkpoint or base model.
            model_name: Human-readable model name.
            gpu_ids: List of GPU IDs to use; None means all available GPUs (0–7).
        """
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.gpu_ids = gpu_ids if gpu_ids else list(range(8))
        self.primary_gpu = self.gpu_ids[0]
        self.engines = []
        self.templates = []
        self.current_engine_idx = 0
        self.lock = threading.Lock()
        self._load_models()

    def _load_models(self):
        """Load the model on one or more GPUs (each model instance uses its own GPU)."""
        print(f'Loading {self.model_name} on GPU(s): {self.gpu_ids}')
        args_json_path = os.path.join(self.checkpoint_path, 'args.json')
        has_args_json = os.path.exists(args_json_path)
        if has_args_json:
            print(f'  Detected checkpoint path (has args.json)')
            args = BaseArguments.from_pretrained(self.checkpoint_path)
            model_path = args.model
            template_name = args.template
            system = args.system
            has_lora = True
        else:
            print(f'  Detected base model path (no args.json), using default config')
            model_path = self.checkpoint_path
            if 'qwen3' in model_path.lower():
                template_name = 'qwen3_vl'
            else:
                template_name = 'qwen2_vl'
            system = None
            has_lora = False
        primary_gpu = self.gpu_ids[0]
        device = f'cuda:{primary_gpu}'
        multi_gpu = len(self.gpu_ids) > 1
        if multi_gpu:
            print(f'  Using GPUs: {self.gpu_ids} (device_map=auto)')
        else:
            print(f'  Using GPU: {primary_gpu} ({device})')
        print(f'  Model path: {model_path}')
        print(f'  Template: {template_name}')
        try:
            max_memory_dict = None
            if multi_gpu and torch.cuda.is_available():
                ratio = float(os.environ.get('VLM_MAX_MEMORY_RATIO', '0.9'))
                max_memory_dict = {}
                for gpu_id in self.gpu_ids:
                    props = torch.cuda.get_device_properties(gpu_id)
                    total_memory_gb = props.total_memory / 1024 ** 3
                    max_memory_gb = int(total_memory_gb * ratio)
                    max_memory_dict[gpu_id] = f'{max_memory_gb}GiB'
                print(f'  max_memory (ratio {ratio:.2f}): {max_memory_dict}')
            model, tokenizer = get_model_tokenizer(model_path, device_map='auto' if multi_gpu else device, max_memory=max_memory_dict if max_memory_dict else None)
            if has_lora:
                print(f'  Loading LoRA adapter from checkpoint...')
                model = Swift.from_pretrained(model, self.checkpoint_path)
            else:
                print(f'  Skipping LoRA adapter (base model path)')
            if torch.cuda.is_available() and (not multi_gpu):
                model = model.to(device)
            template = get_template(template_name, tokenizer, system)
            engine = PtEngine.from_model_template(model, template)
            self.engines.append(engine)
            self.templates.append(template)
            if multi_gpu:
                print(f'{self.model_name} loaded successfully on GPUs {self.gpu_ids}!')
            else:
                print(f'{self.model_name} loaded successfully on GPU {primary_gpu}!')
        except Exception as e:
            print(f'Error loading {self.model_name} on GPU {primary_gpu}: {e}')
            import traceback
            traceback.print_exc()
            raise e

    def _get_next_engine(self):
        """Round-robin selection of the next available engine for load balancing."""
        with self.lock:
            engine = self.engines[self.current_engine_idx]
            template = self.templates[self.current_engine_idx]
            self.current_engine_idx = (self.current_engine_idx + 1) % len(self.engines)
            return (engine, template)

    def infer(self, messages: List[Dict], images: Optional[List[str]]=None, max_new_tokens: int=2048, temperature: float=0.0) -> str:
        """
        Run inference (with automatic load balancing across GPUs).
        
        Args:
            messages: List of chat messages, e.g. [{"role": "user", "content": "..."}].
            images: Optional list of image paths.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
        
        Returns:
            The generated text from the model.
        """
        request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
        processed_messages = []
        for msg in messages:
            content = msg.get('content', '')
            if images and isinstance(content, str):
                image_tag = '<image>'
                existing_tags = content.count(image_tag)
                if existing_tags < len(images) and msg.get('role') == 'user':
                    content = (image_tag + '\n') * (len(images) - existing_tags) + content
            processed_messages.append({**msg, 'content': content})
        infer_request = InferRequest(messages=processed_messages, images=images if images else None)
        engine, template = self._get_next_engine()
        resp_list = engine.infer([infer_request], request_config, template=template)
        response = resp_list[0].choices[0].message.content
        return response

def load_base_model(model_path: str, model_name: str='Base Model', gpu_id: int=3):
    """
    Load a base model without any LoRA adapters.
    
    Args:
        model_path: Base model path.
        model_name: Descriptive model name.
        gpu_id: GPU ID to place the model on.
    
    Returns:
        Dict containing the engine, template, and gpu_id.
    """
    device = f'cuda:{gpu_id}'
    print(f'Loading {model_name} on GPU: {gpu_id} ({device})')
    try:
        model, tokenizer = get_model_tokenizer(model_path, device_map=device)
        if torch.cuda.is_available():
            model = model.to(device)
        template = get_template('qwen2_vl', tokenizer)
        engine = PtEngine.from_model_template(model, template)
        print(f'{model_name} loaded successfully on GPU {gpu_id}!')
        return {'engine': engine, 'template': template, 'gpu_id': gpu_id}
    except Exception as e:
        print(f'Error loading {model_name} on GPU {gpu_id}: {e}')
        import traceback
        traceback.print_exc()
        raise e

class MultimodalAgentService:
    """Multimodal dialogue agent service for VLM A/B/C and image generation."""

    def __init__(self, vlm_a_gpus: Optional[List[int]]=None, vlm_b_gpus: Optional[List[int]]=None, vlm_c_gpus: Optional[List[int]]=None):
        """
        Initialize the service.
        """
        self.debug_logs = []
        self.use_gemini_understanding = os.environ.get('USE_GEMINI_UNDERSTANDING_API', '0') == '1'
        self.use_qwen3_understanding = os.environ.get('USE_QWEN3_UNDERSTANDING', '0') == '1'
        self.qwen3_vl_path = 'Qwen/Qwen3-VL-32B-Instruct'
        self.qwen3_vl = None
        self.gemini_understanding_client = None
        self.gemini_understanding_model_name = None
        self.qwen_base_model = None
        self.base_model_path = os.environ.get('BASE_MODEL_PATH', 'Qwen/Qwen2.5-VL-7B-Instruct')
        self.vlm_a_path = os.environ.get('VLM_A_PATH', './checkpoints/memory_writer')
        self.vlm_b_path = os.environ.get('VLM_B_PATH', './checkpoints/retriever')
        self.vlm_c_path = os.environ.get('VLM_C_PATH', './checkpoints/controller')
        self.qwen_gen_path = os.environ.get('QWEN_GEN_PATH', 'Qwen/Qwen-Image')
        self.qwen_edit_path = os.environ.get('QWEN_EDIT_PATH', 'Qwen/Qwen-Image-Edit-2511')
        self.qwen_lora_path = os.environ.get('QWEN_IMAGE_EDIT_LORA_PATH', None)
        if vlm_a_gpus is None:
            vlm_a_gpus = [0]
        if vlm_b_gpus is None:
            vlm_b_gpus = [1]
        if vlm_c_gpus is None:
            vlm_c_gpus = [2]
        share_model = os.environ.get('VLM_SHARE_MODEL', '0') == '1'
        shared_path = os.environ.get('VLM_SHARED_PATH')
        shared_gpus_env = os.environ.get('VLM_SHARED_GPUS')
        self._shared_vlm = None
        if share_model:
            shared_path = shared_path or self.vlm_a_path
            if shared_gpus_env:
                shared_gpus = [int(x.strip()) for x in shared_gpus_env.split(',') if x.strip()]
            else:
                shared_gpus = vlm_a_gpus
            print(f'Initializing shared VLM (A/B/C) with GPU: {shared_gpus[0]}')
            print(f'Shared VLM path: {shared_path}')
            shared_vlm = VLMModel(shared_path, 'VLM Shared', gpu_ids=shared_gpus)
            self.vlm_a = shared_vlm
            self.vlm_b = shared_vlm
            self.vlm_c = shared_vlm
            self._shared_vlm = shared_vlm
        else:
            print(f'Initializing VLM A with GPU: {vlm_a_gpus[0]}')
            print(f'Initializing VLM B with GPU: {vlm_b_gpus[0]}')
            self.vlm_a = VLMModel(self.vlm_a_path, 'VLM A', gpu_ids=vlm_a_gpus)
            self.vlm_b = VLMModel(self.vlm_b_path, 'VLM B', gpu_ids=vlm_b_gpus)
            if not self.use_qwen3_understanding:
                print(f'Initializing VLM C with GPU: {vlm_c_gpus[0]}')
                self.vlm_c = VLMModel(self.vlm_c_path, 'VLM C', gpu_ids=vlm_c_gpus)
            else:
                print('VLM C loading skipped (using Qwen3-VL-32B as VLM C)')
        if self.use_qwen3_understanding:
            print(f'Initializing Qwen3-VL-32B for understanding from {self.qwen3_vl_path}...')
            qwen3_gpus_env = os.environ.get('QWEN3_GPUS', '0,1,2,3,4,5,6')
            qwen3_gpus = [int(x.strip()) for x in qwen3_gpus_env.split(',') if x.strip()]
            max_memory_dict = {}
            for gid in qwen3_gpus:
                max_memory_dict[gid] = '40GiB'
            try:
                self.qwen3_vl = VLMModel(self.qwen3_vl_path, 'Qwen3-VL-32B', gpu_ids=qwen3_gpus)
                print('Qwen3-VL-32B initialized successfully!')
                self.vlm_c = self.qwen3_vl
            except Exception as e:
                print(f'❌ Critical Error: Failed to initialize Qwen3-VL-32B: {e}')
                import traceback
                traceback.print_exc()
                self.use_qwen3_understanding = False
        if self.use_gemini_understanding:
            from cma_api_client import CompatibleChatClient
            self.gemini_understanding_model_name = os.environ.get('CMA_JUDGE_MODEL', 'gemini-3-pro-preview')
            self.gemini_understanding_client = CompatibleChatClient(model=self.gemini_understanding_model_name)
        if not self.use_gemini_understanding:
            qwen_base_gpus = os.environ.get('QWEN_BASE_GPUS')
            if qwen_base_gpus:
                qwen_base_gpus = [int(x.strip()) for x in qwen_base_gpus.split(',')]
            else:
                qwen_base_gpus = [3]
            use_shared_for_base = os.environ.get('VLM_USE_SHARED_FOR_BASE', '1') == '1'
            if share_model and self._shared_vlm is not None and use_shared_for_base:
                print('Using shared VLM for task classification; skip Qwen2.5VL Base load')
                self.qwen_base_model = {'engine': self._shared_vlm.engines[0], 'template': self._shared_vlm.templates[0]}
            else:
                print(f'Initializing Qwen2.5VL Base Model with GPU: {qwen_base_gpus[0]}')
                self.qwen_base_model = load_base_model(self.base_model_path, 'Qwen2.5VL Base', qwen_base_gpus[0])
        self.image_generator = None
        self.image_editor = None

        def _parse_gpu_list(env_key, default):
            val = os.environ.get(env_key)
            if val:
                return [int(x.strip()) for x in val.split(',') if x.strip()]
            return default
        qwen_gen_gpus = _parse_gpu_list('QWEN_GEN_GPUS', [3, 4, 5, 6, 7])
        qwen_gen_gpu = int(os.environ.get('QWEN_GEN_GPU', str(qwen_gen_gpus[0])))
        qwen_edit_gpus = _parse_gpu_list('QWEN_EDIT_GPUS', [3, 4, 5, 6, 7])
        qwen_edit_gpu = int(os.environ.get('QWEN_EDIT_GPU', str(qwen_edit_gpus[0])))
        try:
            from qwen_lightning_generator import QwenImageGenerator
            print(f'Initializing Qwen-Image (generate) with GPUs: {qwen_gen_gpus}  path: {self.qwen_gen_path}')
            self.image_generator = QwenImageGenerator(model_path=self.qwen_gen_path, gpu_id=qwen_gen_gpu, gpu_ids=qwen_gen_gpus)
            print('Qwen-Image (generate) initialized successfully!')
        except Exception as e:
            print(f'❌ Failed to initialize Qwen-Image (generate): {e}')
            import traceback
            traceback.print_exc()
        try:
            from qwen_lightning_generator import QwenImageEditGenerator
            print(f'Initializing Qwen-Image-Edit (edit) with GPUs: {qwen_edit_gpus}  path: {self.qwen_edit_path}')
            self.image_editor = QwenImageEditGenerator(model_path=self.qwen_edit_path, gpu_id=qwen_edit_gpu, gpu_ids=qwen_edit_gpus)
            print('Qwen-Image-Edit (edit) initialized successfully!')
        except Exception as e:
            print(f'❌ Failed to initialize Qwen-Image-Edit (edit): {e}')
            import traceback
            traceback.print_exc()
        self.gemini_generator = self.image_editor or self.image_generator
        self.sessions: Dict[str, Dict] = {}

    def _log_debug(self, model_name: str, prompt: str, output: str, images: Optional[List[str]]=None):
        """Record a single inference log entry."""
        log_entry = {'model': model_name, 'prompt': prompt, 'output': output, 'images': images, 'timestamp': datetime.now().strftime('%H:%M:%S')}
        self.debug_logs.append(log_entry)
        print(f"\n[DEBUG LOG] {model_name} @ {log_entry['timestamp']}")
        print(f'Prompt: {prompt}')
        print(f'Output: {output}')

    def get_debug_html(self, logs: Optional[List[Dict]]=None) -> str:
        """Generate HTML used for Gradio debug display."""
        target_logs = logs if logs is not None else self.debug_logs
        if not target_logs:
            return ''
        debug_html_str = '<div style="margin-top: 10px; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background-color: #f9f9f9;">'
        debug_html_str += '<details><summary style="cursor: pointer; color: #2196F3; font-weight: bold;">🔍 View full pipeline debug details (PAE / CoRE / MEC / Generator)</summary>'
        for log in target_logs:
            escaped_prompt = html.escape(str(log['prompt']))
            escaped_output = html.escape(str(log['output']))
            debug_html_str += f'<div style="margin-top: 8px; border-bottom: 1px dashed #ccc; padding-bottom: 8px;">'
            debug_html_str += f"""<strong style="color: #e91e63;">[{log['timestamp']}] {log['model']}</strong><br>"""
            debug_html_str += f'<details style="margin-left: 15px;"><summary style="font-size: 0.9em; color: #666; cursor: pointer;">Input Content</summary>'
            debug_html_str += f'<pre style="white-space: pre-wrap; font-size: 0.85em; background: #eee; padding: 5px; border-radius: 4px;">{escaped_prompt}</pre>'
            if log.get('images'):
                debug_html_str += f"""<div style="font-size: 0.8em; color: #888;">Attached Images: {log['images']}</div>"""
            debug_html_str += '</details>'
            debug_html_str += f'<div style="margin-left: 15px; margin-top: 5px;">'
            debug_html_str += f'<span style="font-size: 0.9em; color: #666;">Output Result:</span>'
            debug_html_str += f'<pre style="white-space: pre-wrap; font-size: 0.85em; background: #fff; border: 1px solid #eee; padding: 5px; border-radius: 4px;">{escaped_output}</pre>'
            debug_html_str += '</div></div>'
        debug_html_str += '</details></div>'
        return debug_html_str

    def clear_debug_logs(self):
        """Clear in-memory debug logs."""
        self.debug_logs = []

    def create_session(self, session_id: Optional[str]=None) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional explicit session ID; auto-generated if None.
        
        Returns:
            The created session ID.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        self.sessions[session_id] = {'session_id': session_id, 'created_at': datetime.now().isoformat(), 'memory_bank': MemoryBank(), 'turn_count': 0, 'chat_history': []}
        print(f'Created new session: {session_id}')
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get full session state."""
        return self.sessions.get(session_id)

    def summarize_image(self, session_id: str, image_path: str) -> Dict:
        """
        Summarize an image by calling VLM A (tags + description format, matching SFT data).
        """
        if session_id not in self.sessions:
            raise ValueError(f'Session {session_id} not found')
        session = self.sessions[session_id]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image not found: {image_path}')
        user_message = '<image>Please analyze this image in detail and generate complete image features + description in English.The output must be JSON containing the following fields: tags (tag list, at least 5 tags, array format), description (detailed description, at least 100 words). Please ensure the output is valid JSON.'
        messages = [{'role': 'user', 'content': user_message}]
        print(f"\n{'=' * 80}")
        print(f'[VLM A] Start image summarization (tags + description)')
        print(f'[VLM A] Input image path: {image_path}')
        print(f"{'-' * 80}")
        response = self.vlm_a.infer(messages=messages, images=[image_path], max_new_tokens=512, temperature=0.0)
        self._log_debug('Perceptual Abstraction Engine (PAE)', user_message, response, [image_path])
        print(f'[VLM A] Full output:\n{response}')
        print(f"{'=' * 80}\n")
        memory = None
        try:
            memory = json.loads(response)
        except Exception:
            match = re.search('\\{.*\\}', response or '', re.DOTALL)
            if match:
                try:
                    memory = json.loads(match.group())
                except Exception:
                    memory = None
        tags = memory.get('tags', []) if isinstance(memory, dict) else []
        if not isinstance(tags, list):
            tags = []
        description = memory.get('description', '') if isinstance(memory, dict) else ''
        if not isinstance(description, str):
            description = str(description) if description else ''
        if not tags and (not description):
            description = response or ''
        result_memory = {'tags': tags, 'description': description, 'image_path': image_path}
        print(f'[VLM A] Parsed: tags={len(tags)}, description_length={len(description)}')
        memory_id = session['memory_bank'].add_memory(result_memory)
        session['turn_count'] += 1
        print(f'Image summarized and added to memory bank with ID: {memory_id}')
        return {'memory_id': memory_id, 'memory': result_memory, 'turn_count': session['turn_count']}

    def _add_to_chat_history(self, session_id: str, role: str, content: str, image_summary: Optional[Dict]=None, debug_logs: Optional[List[Dict]]=None):
        """
        Append one turn to the chat history.
        
        Args:
            session_id: Session ID.
            role: "user" or "assistant".
            content: Text content to store.
            image_summary: Optional image summary (title, caption, tags).
            debug_logs: Optional list of debug log entries.
        """
        if session_id not in self.sessions:
            raise ValueError(f'Session {session_id} not found')
        session = self.sessions[session_id]
        turn_count = session['turn_count']
        chat_entry = {'turn': turn_count, 'role': role, 'content': content, 'timestamp': datetime.now().isoformat()}
        if debug_logs:
            chat_entry['debug_logs'] = debug_logs.copy()
        if image_summary:
            chat_entry['image_summary'] = {'title': image_summary.get('title', ''), 'caption': image_summary.get('caption', ''), 'tags': image_summary.get('tags', []), 'image_path': image_summary.get('image_path', '')}
        session['chat_history'].append(chat_entry)

    def _format_chat_history_for_vlm_c(self, session_id: str) -> str:
        """
        Format full chat history into a string input for VLM C.
        """
        if session_id not in self.sessions:
            return ''
        session = self.sessions[session_id]
        chat_history = session['chat_history']
        if not chat_history:
            return ''
        formatted_parts = ['Conversation history:']
        for entry in chat_history:
            role = entry.get('role', '')
            content = entry.get('content', '')
            if isinstance(content, dict):
                content = content.get('prompt', '')
            role_label = 'User' if role == 'user' else 'Assistant'
            formatted_parts.append(f'{role_label}: {content}')
        return '\n'.join(formatted_parts) + '\n'

    def _build_vlm_c_generate_prompt(self, instruction: str) -> str:
        """Build VLM C prompt for text-to-image generation (no retrieved image)."""
        return f"You are a prompt rewriting assistant for text-to-image generation models.\n\nYour task is to convert a short user instruction into a detailed visual prompt suitable for an image generation model.\n\nGuidelines:\n1. Identify the main subject described by the user.\n2. Expand visual attributes such as color, shape, size, and appearance.\n3. Add reasonable scene or environment details if not specified.\n4. Optionally include viewpoint or composition to improve image quality.\n5. Keep the user's original intent unchanged.\n\nThe output should be a single concise image generation prompt.\n\nDo not explain anything.\nOnly output the rewritten prompt.\n\nUser instruction:\n{instruction}\n\nRewrite this instruction into a detailed image generation prompt for a text-to-image model.\n\nOutput only the final prompt."

    def _build_vlm_c_edit_prompt(self, instruction: str) -> str:
        """Build VLM C prompt for image editing (has retrieved image)."""
        return f"You are a multimodal prompt rewriting assistant for image editing models.\n\nYour task is to rewrite a user's editing instruction into a precise image editing prompt.\n\nYou will receive:\n- an image\n- a user editing instruction\n- optional dialogue context\n\nYour goal is to produce a clear editing prompt that an image editing model can follow.\n\nGuidelines:\n1. Identify the object or region to modify using the image.\n2. Clearly describe the modification requested by the user.\n3. Preserve the user's intent without changing the meaning.\n4. Explicitly indicate that all other parts of the image should remain unchanged.\n5. The output must be a single concise editing prompt.\n\nDo not explain anything.\nOnly output the rewritten editing prompt.\n\nImage editing instruction:\n{instruction}\n\nRewrite this instruction into a precise image editing prompt for an image editing model.\nUse the image to resolve ambiguous references if needed.\n\nOutput only the final editing prompt."

    def get_memory_bank(self, session_id: str) -> Dict:
        """Get the memory bank for a given session."""
        if session_id not in self.sessions:
            raise ValueError(f'Session {session_id} not found')
        return self.sessions[session_id]['memory_bank'].to_dict()

    def clear_memory_bank(self, session_id: str):
        """Clear the memory bank for the given session."""
        if session_id not in self.sessions:
            raise ValueError(f'Session {session_id} not found')
        self.sessions[session_id]['memory_bank'].clear()
        print(f'Memory bank cleared for session: {session_id}')

    def detect_task_type(self, user_input: str, image_path: Optional[str]=None) -> str:
        """
        Classify the current turn into a task type using Qwen2.5-VL Base (0114-style) or Gemini.
        """
        if self.use_gemini_understanding and self.gemini_understanding_client:
            try:
                image_info = 'The user has just uploaded an image.' if image_path else 'The user did not upload an image.'
                classification_prompt = f'You are a multimodal task classification expert. \nGiven the user input and current state, categorize the task into one of the following five types:\n\n1. **generate** (text-to-image): the user wants to start a brand new drawing topic, without referencing historical images.\n2. **edit** (single-image edit): the user wants to modify a specific image (usually the latest or a specified one).\n3. **composite** (multi-image composition): the user wants to combine elements or styles from two or more images.\n4. **understand** (image understanding/MMU): the user asks about image content, evaluation or analysis.\n5. **pure_chat** (text-only chat): the user is just greeting, chit-chatting, or asking general knowledge, with no image involved.\n\n**Current state:**\n- User text: "{user_input}"\n- Image status: {image_info}\n\n**Output requirement:**\nReturn JSON only, with no explanation:\n{{"task_type": "generate"|"edit"|"composite"|"understand"|"pure_chat"}}'
                gemini_res = self.gemini_understanding_client.chat(prompt=classification_prompt, images=[image_path] if image_path else None, model=self.gemini_understanding_model_name or 'gemini-3-pro-preview', temperature=0.0)
                response = self.gemini_understanding_client.extract_response_text(gemini_res)
                self._log_debug('Multimodal Executive Controller (MEC) - Task Classifier', classification_prompt, response)
                json_match = re.search('\\{[^}]*"task_type"[^}]*\\}', response, re.IGNORECASE)
                if json_match:
                    result = json.loads(json_match.group())
                    task_type = result.get('task_type', 'understand').lower()
                    if task_type in ['generate', 'edit', 'composite', 'understand', 'pure_chat']:
                        print(f'[Task classification] (Gemini) detected task_type: {task_type}')
                        return task_type
                response_lower = response.lower()
                for t in ['composite', 'pure_chat', 'edit', 'generate', 'understand']:
                    if t in response_lower:
                        return t
                return 'understand'
            except Exception as e:
                print(f'[Task classification] Gemini classifier failed: {e}')
                if not self.qwen_base_model:
                    return 'understand'
        image_info = 'The user has just uploaded an image.' if image_path else 'The user did not upload an image.'
        classification_prompt = f'You are a multimodal task classification expert. \nGiven the user input and current state, categorize the task into one of the following five types:\n\n1. **generate** (text-to-image): the user wants to start a brand new drawing topic, without referencing historical images.\n2. **edit** (single-image edit): the user wants to modify a specific image (usually the latest or a specified one).\n3. **composite** (multi-image composition): the user wants to combine elements or styles from two or more images.\n4. **understand** (image understanding/MMU): the user asks about image content, evaluation or analysis.\n5. **pure_chat** (text-only chat): the user is just greeting, chit-chatting, or asking general knowledge, with no image involved.\n\n**Current state:**\n- User text: "{user_input}"\n- Image status: {image_info}\n\n**Output requirement:**\nReturn JSON only, with no explanation:\n{{"task_type": "generate"|"edit"|"composite"|"understand"|"pure_chat"}}'
        if image_path and '<image>' not in classification_prompt:
            classification_prompt = '<image>' + classification_prompt
        messages = [{'role': 'user', 'content': classification_prompt}]
        input_images = [image_path] if image_path else None
        try:
            request_config = RequestConfig(max_tokens=100, temperature=0.0)
            infer_request = InferRequest(messages=messages, images=input_images if input_images else None)
            engine = self.qwen_base_model['engine']
            template = self.qwen_base_model['template']
            resp_list = engine.infer([infer_request], request_config, template=template)
            response = resp_list[0].choices[0].message.content
            self._log_debug('Multimodal Executive Controller (MEC) - Task Classifier', classification_prompt, response)
            json_match = re.search('\\{[^}]*"task_type"[^}]*\\}', response, re.IGNORECASE)
            if json_match:
                result = json.loads(json_match.group())
                task_type = result.get('task_type', 'understand').lower()
                if task_type in ['generate', 'edit', 'composite', 'understand', 'pure_chat']:
                    print(f'[Task classification] Detected task_type: {task_type}')
                    return task_type
            response_lower = response.lower()
            for t in ['composite', 'pure_chat', 'edit', 'generate', 'understand']:
                if t in response_lower:
                    return t
            return 'understand'
        except Exception as e:
            print(f'[Task classification] Fallback classifier failed: {e}')
            return 'understand'

    def retrieve_relevant_images(self, session_id: str, user_query: str) -> List[int]:
        """
        Use VLM B to retrieve the most relevant image IDs from the memory bank.
        Matches the benchmark SFT format: tags + description text + 448 thumbnails.
        """
        if session_id not in self.sessions:
            raise ValueError(f'Session {session_id} not found')
        session = self.sessions[session_id]
        memory_bank = session['memory_bank']
        all_memories = memory_bank.get_all_memories()
        if not all_memories:
            return []
        max_desc_chars = 300
        candidates_lines = []
        memory_lines = []
        images = []
        for mem in all_memories:
            mem_id = mem.get('id', 0)
            tags = mem.get('tags', [])
            description = mem.get('description', '')
            candidates_lines.append(f'- rank {mem_id}: <image>')
            if isinstance(tags, list):
                tags_text = ', '.join([str(t) for t in tags if t is not None])
            else:
                tags_text = str(tags) if tags else ''
            desc = description or ''
            if max_desc_chars and len(desc) > max_desc_chars:
                desc = desc[:max_desc_chars].rstrip() + '...'
            parts = [f'[{mem_id}]']
            if tags_text:
                parts.append(f'tags: {tags_text}')
            if desc:
                parts.append(f'description: {desc}')
            memory_lines.append(' | '.join(parts))
            img_path = mem.get('image_path')
            if img_path and os.path.exists(img_path):
                images.append(_make_thumbnail(img_path, 448))
        candidates = '\n'.join(candidates_lines)
        memory_text = '\n'.join(memory_lines)
        user_content = f'Candidate image list (aligned with the order in `images`):\n{candidates}\n\nMemory bank information (including tags + description):\n{memory_text}\n\nUser query: {user_query}\n\nPlease select the most relevant image ID list based on the user query and memory bank information. If no historical image is needed, output an empty list.'
        print(f"\n{'=' * 80}")
        print(f'[VLM B] Start image retrieval (tags+desc+448 thumbnails)')
        print(f'[VLM B] User query: {user_query}')
        print(f'[VLM B] Candidates: {len(all_memories)}, Thumbnails: {len(images)}')
        print(f"{'-' * 80}")
        print(f'[VLM B] Input prompt:\n{user_content}')
        print(f"{'-' * 80}")
        messages = [{'role': 'user', 'content': user_content}]
        response = self.vlm_b.infer(messages=messages, images=images if images else None, max_new_tokens=256, temperature=0.0)
        self._log_debug('Cognitive Retrieval Engine (CoRE)', user_content, response, images)
        print(f'[VLM B] Full output:\n{response}')
        print(f"{'=' * 80}\n")
        ranks = self._parse_vlmb_output(response)
        valid_ids = {m.get('id') for m in all_memories}
        seen = set()
        filtered = []
        for r in ranks:
            if r not in seen and r in valid_ids:
                seen.add(r)
                filtered.append(r)
        print(f'Retrieved image IDs: {filtered}')
        return filtered

    @staticmethod
    def _parse_vlmb_output(text: str) -> List[int]:
        """Parse VLM B output into a list of rank integers (matches benchmark logic)."""
        raw = (text or '').strip()
        if not raw:
            return []
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return [int(x) for x in obj if str(x).strip().lstrip('-').isdigit()]
            if isinstance(obj, dict):
                sel = obj.get('selected_index', obj.get('selected_indices', obj.get('ranks')))
                if sel is None:
                    return []
                if isinstance(sel, list):
                    return [int(x) for x in sel if str(x).strip().lstrip('-').isdigit()]
                if isinstance(sel, (int, float)) and int(sel) >= 0:
                    return [int(sel)]
        except Exception:
            pass
        m = re.search('\\[[^\\]]*\\]', raw)
        if m:
            try:
                obj = json.loads(m.group())
                if isinstance(obj, list):
                    return [int(x) for x in obj if str(x).strip().lstrip('-').isdigit()]
            except Exception:
                pass
        nums = re.findall('\\d+', raw)
        return [int(n) for n in nums] if nums else []

    def _strip_vlm_c_tags(self, text: str) -> str:
        """
        Strip all VLM C tag markers while keeping the raw text (supports 235b suffix).
        """
        import re
        cleaned = re.sub('<(?:think|ans)(?:235b)?>|</(?:think|ans)(?:235b)?>', '', text, flags=re.I).strip()
        return cleaned

    def _extract_vlm_c_ans(self, text: str) -> str:
        """
        Extract the <ans> or <ans235b> section from VLM C output (for final answers / image prompts).
        """
        ans_match = re.search('<(?:ans|ans235b)>(.*?)</(?:ans|ans235b)>', text, re.S | re.I)
        if ans_match:
            return ans_match.group(1).strip()
        return self._strip_vlm_c_tags(text)

    def _format_vlm_c_output_for_ui(self, text: str) -> str:
        """
        Format raw VLM C output into styled HTML, separating thought and answer (supports 235b suffix).
        """
        import re
        lower_text = text.lower()
        think_content = ''
        think_patterns = [('<think235b>', '</think235b>'), ('<think>', '</think>')]
        for start_tag, end_tag in think_patterns:
            first_think_match = re.search(start_tag, lower_text)
            if first_think_match:
                start_pos = first_think_match.end()
                last_think_close_match = re.search(end_tag, lower_text[start_pos:])
                if last_think_close_match:
                    think_content = text[start_pos:start_pos + last_think_close_match.start()].strip()
                else:
                    ans_start_match = re.search('<(?:ans|ans235b)>', lower_text[start_pos:])
                    if ans_start_match:
                        think_content = text[start_pos:start_pos + ans_start_match.start()].strip()
                    else:
                        think_content = text[start_pos:].strip()
                break
        ans_content = ''
        ans_patterns = [('<ans235b>', '</ans235b>'), ('<ans>', '</ans>')]
        for start_tag, end_tag in ans_patterns:
            last_ans_matches = list(re.finditer(start_tag, lower_text))
            if last_ans_matches:
                last_ans_match = last_ans_matches[-1]
                start_pos = last_ans_match.end()
                last_ans_close_match = re.search(end_tag, lower_text[start_pos:])
                if last_ans_close_match:
                    ans_content = text[start_pos:start_pos + last_ans_close_match.start()].strip()
                else:
                    ans_content = text[start_pos:].strip()
                break
        if not ans_content:
            last_think_close_match = re.search('</(?:think|think235b)>', lower_text)
            if last_think_close_match:
                ans_content = text[last_think_close_match.end():].strip()
            elif not think_content:
                ans_content = self._strip_vlm_c_tags(text)
        formatted = ''
        if think_content:
            display_think = self._strip_vlm_c_tags(think_content)
            if 'thought process omitted' not in display_think.lower():
                formatted += f'<div style="background-color: #f7f7f7; border-left: 4px solid #9e9e9e; padding: 8px 12px; margin-bottom: 10px; font-style: italic; color: #616161; font-size: 0.95em;">'
                formatted += f'<div style="font-weight: bold; margin-bottom: 4px; font-size: 0.85em; color: #757575;">🤔 Thought process:</div>'
                formatted += f'{display_think}</div>'
        if ans_content:
            display_ans = self._strip_vlm_c_tags(ans_content)
            if display_ans:
                formatted += f'<div>{display_ans}</div>'
        return formatted

    def generate_image(self, session_id: str, user_instruction: str, action_type: str='edit', reference_image_ids: Optional[List[int]]=None, use_vlm_b_retrieval: bool=True, save_dir: Optional[str]=None) -> Dict:
        """
        Full image generation pipeline (aligned with the 0114 multi-image retrieval logic).
        """
        if session_id not in self.sessions:
            raise ValueError(f'Session {session_id} not found')
        if self.image_generator is None and self.image_editor is None:
            raise ValueError('No image generator or editor initialized.')
        session = self.sessions[session_id]
        memory_bank = session['memory_bank']
        all_memories = memory_bank.get_all_memories()
        retrieved_image_ids = reference_image_ids if reference_image_ids else []
        if use_vlm_b_retrieval and (not reference_image_ids) and (len(all_memories) > 0):
            print(f'[VLM B] Running retrieval...')
            retrieved_image_ids = self.retrieve_relevant_images(session_id, user_instruction)
        retrieved_memories = []
        input_img_paths = []
        for mem_id in retrieved_image_ids:
            mem = memory_bank.get_memory(mem_id)
            if mem:
                retrieved_memories.append({'id': mem_id, 'memory': mem})
                if mem.get('image_path'):
                    input_img_paths.append(mem['image_path'])
        if not input_img_paths:
            from PIL import Image
            dummy_path = '/tmp/agent_dummy_black.png'
            if not os.path.exists(dummy_path):
                Image.new('RGB', (1024, 1024), (0, 0, 0)).save(dummy_path)
            input_img_paths.append(dummy_path)
        self._add_to_chat_history(session_id, 'user', user_instruction)
        has_retrieved_images = len(retrieved_image_ids) > 0
        if has_retrieved_images:
            vlm_c_input = self._build_vlm_c_edit_prompt(instruction=user_instruction)
            prompt_mode = 'edit'
        else:
            vlm_c_input = self._build_vlm_c_generate_prompt(instruction=user_instruction)
            prompt_mode = 'generate'
        print(f'[VLM C] Prompt mode: {prompt_mode} | Start inference with {len(input_img_paths)} image(s)...')
        messages = [{'role': 'user', 'content': vlm_c_input}]
        raw_generation_output = self.vlm_c.infer(messages=messages, images=input_img_paths if input_img_paths else None, max_new_tokens=256)
        agent_prompt = raw_generation_output.strip()
        if agent_prompt:
            combined_generation_prompt = agent_prompt
        elif has_retrieved_images:
            combined_generation_prompt = f'{user_instruction}\nKeep everything else in the image unchanged.'
        else:
            combined_generation_prompt = user_instruction
        self._log_debug(f"Multimodal Executive Controller (MEC) - {('Edit Prompt Rewriter' if has_retrieved_images else 'Generate Prompt Writer')}", vlm_c_input, raw_generation_output, input_img_paths)
        if has_retrieved_images:
            active_gen = self.image_editor or self.image_generator
            generator_name = 'Qwen-Image-Edit (Local)'
            if active_gen is None:
                raise ValueError('No image editor available. Check Qwen-Image-Edit initialization.')
            print(f'[Generator] Calling {generator_name} to edit image...')
            gen_result = active_gen.generate(prompt=combined_generation_prompt, images=input_img_paths, cid=session_id)
        else:
            active_gen = self.image_generator or self.image_editor
            generator_name = 'Qwen-Image (Local)'
            if active_gen is None:
                raise ValueError('No image generator available. Check Qwen-Image initialization.')
            print(f'[Generator] Calling {generator_name} to generate image...')
            gen_result = active_gen.generate(prompt=combined_generation_prompt, images=None, cid=session_id)
        log_info = gen_result.get('data', {})
        log_path = log_info.get('local_path') or log_info.get('url') or 'failed'
        self._log_debug(f'{generator_name} (Generator)', combined_generation_prompt, f'Image generated: {log_path}', input_img_paths)
        local_path = gen_result.get('data', {}).get('local_path')
        image_url = gen_result.get('data', {}).get('url') or (gen_result.get('data', {}).get('urls', [None])[0] if gen_result.get('data', {}).get('urls') else None)
        if not local_path and image_url:
            import requests
            if save_dir is None:
                save_dir = '/tmp/agent_generated'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            final_path = os.path.join(save_dir, f"{session_id}_turn_{session['turn_count']}_gen.jpg")
            try:
                response = requests.get(image_url, timeout=60)
                response.raise_for_status()
                with open(final_path, 'wb') as f:
                    f.write(response.content)
                print(f'Downloaded image from Gemini API to: {final_path}')
            except Exception as e:
                raise ValueError(f'Failed to download image from Gemini API: {e}')
        elif local_path and os.path.exists(local_path):
            if save_dir is None:
                save_dir = '/tmp/agent_generated'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            final_path = os.path.join(save_dir, f"{session_id}_turn_{session['turn_count']}_gen.jpg")
            import shutil
            shutil.copy(local_path, final_path)
        else:
            raise ValueError(f'Generation failed: {gen_result}')
        summary_res = self.summarize_image(session_id, final_path)
        if has_retrieved_images:
            assistant_reply = '✅ Edit succeeded!'
        else:
            assistant_reply = '✅ Generate succeeded!'
        self._add_to_chat_history(session_id, 'assistant', assistant_reply, image_summary=summary_res['memory'], debug_logs=self.debug_logs)
        return {'image_path': final_path, 'raw_vlm_c_output': raw_generation_output, 'generation_prompt': combined_generation_prompt, 'retrieved_image_ids': retrieved_image_ids, 'memory': summary_res['memory']}

    def _format_memory_for_vlm_c(self, memory: Dict, mem_id: Optional[int]=None) -> str:
        """Format one memory entry into VLM C input text (tags + description)."""
        tags = memory.get('tags', [])
        if isinstance(tags, list):
            tags_text = ', '.join([str(t) for t in tags if t is not None])
        else:
            tags_text = str(tags) if tags else ''
        description = memory.get('description', '')
        parts = []
        if mem_id is not None:
            parts.append(f'Memory[{mem_id}]')
        if tags_text:
            parts.append(f'tags: {tags_text}')
        if description:
            desc = description[:300].rstrip() + '...' if len(description) > 300 else description
            parts.append(f'description: {desc}')
        return ' | '.join(parts) if parts else '(empty memory)'

    def understand_task(self, session_id: str, user_instruction: str, image_path: Optional[str]=None) -> Dict:
        """
        MMU-style understanding flow: use VLM A/B/C and finally answer via VLM C / Qwen3 / Gemini / Qwen2.5.
        
        Args:
            session_id: Session ID.
            user_instruction: User instruction or question.
            image_path: Optional image path; if provided, we summarize it via VLM A first.
        
        Returns:
            Dict containing answer text, raw outputs, and related image info.
        """
        if session_id not in self.sessions:
            raise ValueError(f'Session {session_id} not found')
        session = self.sessions[session_id]
        memory_bank = session['memory_bank']
        uploaded_memory_id = None
        uploaded_memory = None
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'Image not found: {image_path}')
            print(f'Step 1: Summarizing uploaded image with VLM A...')
            upload_summary = self.summarize_image(session_id, image_path)
            uploaded_memory_id = upload_summary['memory_id']
            uploaded_memory = upload_summary['memory']
            upload_mem = memory_bank.get_memory(uploaded_memory_id)
            if upload_mem:
                upload_mem['image_path'] = image_path
        turn_count = session['turn_count']
        all_memories = memory_bank.get_all_memories()
        retrieved_image_ids = []
        if len(all_memories) > 0:
            print(f'Step 2: Retrieving relevant images with VLM B (turn {turn_count + 1}, Memory Bank has {len(all_memories)} images)...')
            retrieved_image_ids = self.retrieve_relevant_images(session_id, user_instruction)
            print(f'Retrieved image IDs: {retrieved_image_ids}')
        else:
            print(f'Step 2: Skip VLM B retrieval (empty Memory Bank, no historical images).')
        retrieved_memories = []
        if retrieved_image_ids:
            for mem_id in retrieved_image_ids:
                mem = memory_bank.get_memory(mem_id)
                if mem:
                    retrieved_memories.append({'id': mem_id, 'memory': mem})
        input_images_for_check = []
        for retrieved_mem in retrieved_memories:
            mem = retrieved_mem.get('memory', {})
            if mem.get('image_path'):
                input_images_for_check.append(mem['image_path'])
        if image_path:
            input_images_for_check.append(image_path)
        is_text_only = len(input_images_for_check) == 0
        try:
            if is_text_only:
                print(f'Step 3: Text-only (pure chat) mode, no context provided to VLM C...')
                vlm_c_input = f'Current user instruction: {user_instruction}\nTask type: pure_chat\nPlease answer directly according to the instruction. Use <think>...</think> for your reasoning, and use <ans>...</ans> for the final answer. You must think and respond in English only.'
            else:
                print(f'Step 3: Contextual mode (uploaded or retrieved), providing full context to VLM C...')
                image_tags = ''.join([f'<image>\n'] * len(input_images_for_check))
                history_str = self._format_chat_history_for_vlm_c(session_id)
                mem_texts = []
                if retrieved_memories:
                    for rm in retrieved_memories:
                        mem_texts.append(self._format_memory_for_vlm_c(rm['memory'], rm['id']))
                mem_context = 'Retrieved memories:\n' + '\n'.join(mem_texts) + '\n\n' if mem_texts else ''
                vlm_c_input = f'{image_tags}\n{history_str}\n{mem_context}Current user instruction: {user_instruction}\nTask type: mmu\nPlease reason carefully based on the instruction and the reference images, and provide a detailed answer. Use <think>...</think> for your reasoning, and use <ans>...</ans> for the final answer. You must think and respond entirely in English; do not output Chinese.'
            if image_path and uploaded_memory:
                self._add_to_chat_history(session_id, 'user', user_instruction, image_summary=uploaded_memory)
            else:
                self._add_to_chat_history(session_id, 'user', user_instruction)
            messages = [{'role': 'user', 'content': vlm_c_input}]
            if self.use_qwen3_understanding and self.qwen3_vl:
                raw_answer_output = self.qwen3_vl.infer(messages=messages, images=input_images_for_check if input_images_for_check else None, max_new_tokens=1024, temperature=0.0)
                formatted_answer = self._format_vlm_c_output_for_ui(raw_answer_output)
                self._log_debug('Multimodal Executive Controller (MEC) - MMU Answer', vlm_c_input, raw_answer_output, input_images_for_check)
                print(f'Generated answer via Qwen3-VL-32B: {formatted_answer[:100]}...')
            elif self.use_gemini_understanding and self.gemini_understanding_client:
                raw_answer_output = self.gemini_understanding_client.chat(prompt=vlm_c_input, images=input_images_for_check if input_images_for_check else None, model=self.gemini_understanding_model_name or 'gemini-3-pro-preview', temperature=0.0)
                response_text = self.gemini_understanding_client.extract_response_text(raw_answer_output)
                formatted_answer = self._format_vlm_c_output_for_ui(response_text)
                self._log_debug('Multimodal Executive Controller (MEC) - MMU Answer', vlm_c_input, response_text, input_images_for_check)
                print(f'Generated answer via Gemini Understanding API: {formatted_answer[:100]}...')
            else:
                raw_answer_output = self.vlm_c.infer(messages=messages, images=input_images_for_check if input_images_for_check else None, max_new_tokens=1024, temperature=0.0)
                formatted_answer = self._format_vlm_c_output_for_ui(raw_answer_output)
                self._log_debug('Multimodal Executive Controller (MEC) - MMU Answer', vlm_c_input, raw_answer_output, input_images_for_check)
                print(f'Generated answer via VLM C: {formatted_answer[:100]}...')
            answer_text = self._extract_vlm_c_ans(raw_answer_output if isinstance(raw_answer_output, str) else str(raw_answer_output))
            if not answer_text:
                answer_text = self._strip_vlm_c_tags(raw_answer_output if isinstance(raw_answer_output, str) else str(raw_answer_output))
            self._add_to_chat_history(session_id, 'assistant', answer_text, debug_logs=self.debug_logs)
            return {'answer': answer_text, 'raw_output': raw_answer_output, 'retrieved_image_ids': retrieved_image_ids, 'uploaded_memory_id': uploaded_memory_id, 'input_images_count': len(input_images_for_check), 'turn_count': session['turn_count']}
        except Exception as e:
            print(f'Error in understand_task: {e}')
            import traceback
            traceback.print_exc()
            raise e
if __name__ == '__main__':
    service = MultimodalAgentService()
    session_id = service.create_session()
    print('Multimodal Agent Service initialized successfully!')
    print(f'Example session ID: {session_id}')
