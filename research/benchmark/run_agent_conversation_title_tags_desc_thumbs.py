"""
run_agent_conversation 的变体：
VLM B 检索使用 title + tags + description，并附带所有历史图片缩略图（448x448）
"""
import sys
import uuid
from pathlib import Path
from typing import List
BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent / 'ms-swift'
sys.path.insert(0, str(PROJECT_ROOT))
import multimodal_agent_service as mas

def _build_thumbnails(image_paths: List[str], size: int=448) -> List[str]:
    """将输入图片列表生成固定尺寸缩略图，返回缩略图路径列表"""
    if not image_paths:
        return []
    thumb_paths: List[str] = []
    for idx, path in enumerate(image_paths):
        try:
            from PIL import Image
            if not path or not Path(path).exists():
                continue
            with Image.open(path) as img:
                img = img.convert('RGB')
                if img.size != (size, size):
                    img = img.resize((size, size), resample=Image.LANCZOS)
                thumb_path = f'/tmp/agent_vlm_b_thumb_{uuid.uuid4().hex}_{idx}.png'
                img.save(thumb_path)
                thumb_paths.append(thumb_path)
        except Exception as e:
            print(f'⚠️  缩略图生成失败，跳过: {path} ({e})')
            continue
    return thumb_paths

def retrieve_relevant_images_title_tags_desc_thumbs(self, session_id: str, user_query: str) -> List[int]:
    """
    使用 VLM B 从 memory bank 中检索与用户查询最相关的图像 ID
    这里使用 title + tags + description 作为记忆文本，
    并附带所有历史图片的缩略图（固定 448x448）
    """
    if session_id not in self.sessions:
        raise ValueError(f'Session {session_id} not found')
    session = self.sessions[session_id]
    memory_bank = session['memory_bank']
    all_memories = memory_bank.get_all_memories()
    if not all_memories:
        return []
    memory_texts = []
    image_paths = []
    for mem in all_memories:
        mem_id = mem.get('id', '')
        title = mem.get('title') or mem.get('Title') or ''
        tags = mem.get('tags') or mem.get('Tags') or []
        description = mem.get('description') or mem.get('Description') or ''
        if isinstance(tags, list):
            tags_text = ','.join([str(t) for t in tags if t is not None])
        else:
            tags_text = str(tags) if tags is not None else ''
        mem_desc = f'记忆[{mem_id}]: {title}'
        if tags_text:
            mem_desc += f' - {tags_text}'
        if description:
            mem_desc += f' - {description}'
        memory_texts.append(mem_desc)
        image_path = mem.get('image_path')
        if image_path:
            image_paths.append(image_path)
    memory_context = '\n'.join(memory_texts) if memory_texts else '（当前无历史记忆）'
    user_content = f'基于以下记忆库信息：\n{memory_context}\n\n用户查询：{user_query}\n\n请根据用户查询和记忆库信息，选择最相关的图像ID列表。如果不需要引用历史图片，请输出空列表。'
    print(f"\n{'=' * 80}")
    print(f'[VLM B] 开始图像检索 (title+tags+description + thumbnails)')
    print(f"{'=' * 80}")
    print(f'[VLM B] 用户查询: {user_query}')
    print(f'[VLM B] Memory Bank信息:')
    for mem in all_memories:
        mem_id = mem.get('id', '')
        title = mem.get('title') or mem.get('Title') or ''
        tags = mem.get('tags') or mem.get('Tags') or []
        description = mem.get('description') or mem.get('Description') or ''
        print(f'  - Memory ID {mem_id}: {title} - {tags} - {description}')
    print(f"{'-' * 80}")
    print(f'[VLM B] 输入Prompt:')
    print(f'{user_content}')
    print(f"{'-' * 80}")
    messages = [{'role': 'user', 'content': user_content}]
    thumb_paths = _build_thumbnails(image_paths, size=448)
    response = self.vlm_b.infer(messages=messages, images=thumb_paths if thumb_paths else None, max_new_tokens=64, temperature=0.0)
    self._log_debug('VLM B (Retriever title+tags+description+thumbs)', user_content, response, thumb_paths)
    print(f'[VLM B] 完整输出:')
    print(f'{response}')
    print(f"{'=' * 80}\n")
    import re
    numbers = re.findall('\\d+', response)
    if numbers:
        return [int(n) for n in numbers]
    return []
mas.MultimodalAgentService.retrieve_relevant_images = retrieve_relevant_images_title_tags_desc_thumbs
import run_agent_conversation as base
if __name__ == '__main__':
    sys.exit(base.main())
