"""Public chat-completions adapter. Configure a compatible provider explicitly."""
import base64
import mimetypes
import os
from io import BytesIO
from pathlib import Path
import requests

class CompatibleChatClient:
    def __init__(self, api_base_url=None, api_key=None, model=None, **compat):
        self.base_url = (api_base_url or os.environ.get("CMA_API_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or compat.get("app_secret") or os.environ.get("CMA_API_KEY", "")
        self.model = model or os.environ.get("CMA_JUDGE_MODEL", "gemini-3-pro-preview")

    @staticmethod
    def extract_response_text(result):
        if isinstance(result, str):
            return result
        content = result["choices"][0]["message"].get("content", "")
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""

    @staticmethod
    def image_url(image):
        if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
            return image
        if hasattr(image, "save"):
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            data, mime = buffer.getvalue(), "image/png"
        elif isinstance(image, bytes):
            data, mime = image, "image/png"
        else:
            path = Path(str(image).removeprefix("file://"))
            data, mime = path.read_bytes(), mimetypes.guess_type(str(path))[0] or "image/png"
        return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")

    def chat(self, prompt=None, text=None, images=None, model=None, temperature=0.2, max_tokens=None, **kwargs):
        if not self.base_url:
            raise ValueError("Set CMA_API_BASE_URL to a chat-completions-compatible endpoint")
        content = [{"type": "text", "text": prompt if prompt is not None else (text or "")}]
        for image in images or []:
            content.append({"type": "image_url", "image_url": {"url": self.image_url(image)}})
        payload = {"model": model or self.model, "messages": [{"role": "user", "content": content}], "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        url = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()

class GeminiJudge(CompatibleChatClient):
    def chat(self, *args, **kwargs):
        return self.extract_response_text(super().chat(*args, **kwargs))
