"""Portable scenario-engine clients, without the original private gateway."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ms-swift"))
from cma_api_client import CompatibleChatClient

class GeminiUnderstandingClient(CompatibleChatClient):
    def chat(self, text=None, **kwargs):
        return self.extract_response_text(super().chat(text=text, **kwargs))

class GeminiImageGenerator:
    def __init__(self, *args, **kwargs):
        self.generator = None
        self.editor = None

    def generate(self, prompt, images=None, image_format="base64", **kwargs):
        from qwen_lightning_generator import QwenImageGenerator, QwenImageEditGenerator
        if images:
            if self.editor is None:
                self.editor = QwenImageEditGenerator(model_path=os.environ.get("QWEN_EDIT_PATH", "Qwen/Qwen-Image-Edit-2511"), gpu_id=int(os.environ.get("QWEN_EDIT_GPU", "0")))
            engine = self.editor
        else:
            if self.generator is None:
                self.generator = QwenImageGenerator(model_path=os.environ.get("QWEN_GEN_PATH", "Qwen/Qwen-Image"), gpu_id=int(os.environ.get("QWEN_GEN_GPU", "0")))
            engine = self.generator
        result = engine.generate(prompt=prompt, images=images)
        local_path = result.get("data", {}).get("local_path")
        if not local_path:
            raise RuntimeError("Local image backend did not return a generated file")
        return Path(local_path).read_bytes()
