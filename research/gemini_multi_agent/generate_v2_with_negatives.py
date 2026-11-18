import os
import json
import time
import random
import re
from typing import List, Dict, Optional
from multiprocessing import Pool, Manager, cpu_count
from api_clients import GeminiUnderstandingClient, GeminiImageGenerator
_shared_lock = None

class SessionManager:

    def __init__(self, session_id: str, data_dir: str):
        self.session_id = session_id
        self.data_dir = data_dir
        self.memory_bank = []
        self.next_id = 0
        self.conversation_log = []
        self.understand_client = GeminiUnderstandingClient()
        self.generate_client = GeminiImageGenerator()
        self.data_vlm_a, self.data_vlm_b, self.data_vlm_c = ([], [], [])
        self.initial_topics = ['赛博朋克都市', '江南水乡', '森林深处', '宫崎骏风格', '极简建筑', '星际航行', '卡通动物', '复古油画', '超现实梦境', '中国神话']

    def save_image(self, img_bytes: bytes, turn_id: int, suffix: str='') -> str:
        img_name = f"sess_{self.session_id}_turn_{turn_id}{('_' + suffix if suffix else '')}.jpg"
        img_path = os.path.join(self.data_dir, 'images', img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        return img_path

    def generate_user_uploaded_image(self, turn_id: int) -> Optional[str]:
        prompt = f'A high-quality realistic photo or concept art related to {random.choice(self.initial_topics)}.'
        try:
            img_bytes = self.generate_client.generate(prompt=prompt, images=None, image_format='base64')
            if not img_bytes:
                return None
            return self.save_image(img_bytes, turn_id, suffix='user_upload')
        except:
            return None

    def create_memory(self, img_path: str, is_user_uploaded: bool=False) -> Dict:
        """为图片生成摘要 (VLM A 训练数据产生点)"""
        memory_id = self.next_id
        self.next_id += 1
        user_message = '<image>请详细分析这张图片，并以中文生成完整的摘要信息。输出必须是包含以下字段的 JSON 格式：title (标题), description (详细描述), caption (简短说明), tags (标签列表)。'
        response_text = self.understand_client.chat(user_message, images=[img_path], temperature=0.1)
        try:
            clean_json = re.search('\\{.*\\}', response_text, re.DOTALL).group()
            mem_data = json.loads(clean_json)
        except:
            mem_data = {'title': '未知图片', 'description': response_text[:200], 'caption': '图片解析失败', 'tags': []}
        print(f"    [Session {self.session_id}] VLM A Output (ID {memory_id}): {mem_data.get('title')} | {mem_data.get('caption')}")
        self.data_vlm_a.append({'image_path': img_path, 'instruction': user_message, 'output': json.dumps(mem_data, ensure_ascii=False)})
        return {'id': memory_id, 'path': img_path, 'memory': mem_data}

    def user_simulator(self, turn_id: int) -> Dict:
        actions = ['generate', 'edit', 'composite', 'understand', 'pure_chat']
        weights = [0.35, 0.25, 0.0, 0.25, 0.15] if turn_id == 0 else [0.15, 0.4, 0.15, 0.2, 0.1]
        chosen_action = random.choices(actions, weights=weights)[0]
        history_images = []
        for idx, m in enumerate(self.memory_bank):
            ordinal = '第一张' if idx == 0 else f'第{idx + 1}张'
            if idx == len(self.memory_bank) - 1:
                ordinal = '最后一张（刚生成的）'
            history_images.append(f"{ordinal} (程序ID {m['id']}): {m['memory'].get('title')} - {m['memory'].get('caption')}")
        memory_summary = '\n'.join(history_images) if history_images else '（当前无历史图片）'
        user_prompt = f"""你是一个真实的人类用户，正在与AI绘图助手聊天。\n当前任务：{chosen_action}\n\n历史记录中的图片：\n{memory_summary}\n\n【特殊干扰训练要求】：\n如果你被分配的任务是 'generate'（开启新主题），你必须尝试在指令中提到历史图片（如：“那天的那张图不错”、“之前的那个场景让我想起了...”），但你必须在指令的后半部分明确表达你要“抛弃之前的参考，画一个完全不相关的新东西”。\n这种情况下，你的 ref_image_ids 必须设为 []。\n\n【核心指令】：\n请根据任务类型说一句话。\n1. **严禁**在回复中使用“ID”、“程序ID”或任何数字编号来指代图片（除非是“第一张”、“第二张”这种自然口语）。\n2. **指代方式**：使用视觉特征（如“那个有月亮的图”）、相对位置（如“最后那张”）或模糊代词。语气要自然随性。\n\n请输出 JSON：\n{{\n    "instruction": "你的自然语言指令（严禁出现'ID'字样）",\n    "ref_image_ids": [你引用的程序ID列表（从上面提示中获取），不引用则为 []],\n    "should_upload_new_image": bool\n}}\n"""
        response = self.understand_client.chat(user_prompt, temperature=0.7)
        try:
            clean_json = re.search('\\{.*\\}', response, re.DOTALL).group()
            data = json.loads(clean_json)
            if turn_id == 0 and chosen_action in ['edit', 'understand']:
                data['should_upload_new_image'] = True
            return {'action': chosen_action, 'instruction': data.get('instruction', '继续'), 'gt_ids': data.get('ref_image_ids', []), 'upload': data.get('should_upload_new_image', False)}
        except:
            return {'action': 'pure_chat', 'instruction': '你好', 'gt_ids': [], 'upload': False}

    def run_turn(self, turn_id: int):
        user_sim = self.user_simulator(turn_id)
        action, instr, gt_ids = (user_sim['action'], user_sim['instruction'], user_sim['gt_ids'])
        print(f'\n    [Session {self.session_id}] Turn {turn_id} | Action: {action}')
        print(f'    [Session {self.session_id}] User: {instr}')
        input_img_paths = [m['path'] for m in self.memory_bank if m['id'] in gt_ids]
        if user_sim['upload']:
            u_path = self.generate_user_uploaded_image(turn_id)
            if u_path:
                m_item = self.create_memory(u_path, is_user_uploaded=True)
                self.memory_bank.append(m_item)
                u_id = m_item['id']
                print(f'    [Session {self.session_id}] User Uploaded (ID {u_id}): {u_path}')
                if u_id not in gt_ids:
                    gt_ids.append(u_id)
                    input_img_paths.append(u_path)
        mem_texts_for_b = []
        for m in self.memory_bank:
            mid = m['id']
            actual_mem = m['memory']
            t = actual_mem.get('title') or actual_mem.get('Title') or ''
            c = actual_mem.get('caption') or actual_mem.get('Caption') or ''
            mem_desc = f'记忆[{mid}]: {t}'
            if c:
                mem_desc += f' - {c}'
            mem_texts_for_b.append(mem_desc)
        vlm_b_context = '\n'.join(mem_texts_for_b) if mem_texts_for_b else '（当前无历史记忆）'
        vlm_b_prompt = f'基于以下记忆库信息：\n{vlm_b_context}\n\n用户查询：{instr}\n\n请根据用户查询和记忆库信息，选择最相关的图像ID列表。如果不需要引用历史图片，请输出空列表。'
        self.data_vlm_b.append({'instruction': vlm_b_prompt, 'output': f'图像ID: {gt_ids}'})
        print(f'    [Session {self.session_id}] VLM B Output (Retrieval): {gt_ids}')
        if not self.conversation_log:
            history_str = ''
        else:
            history_lines = ['对话历史：']
            for turn in self.conversation_log[-6:]:
                role_label = '用户' if turn['role'] == 'user' else '助手'
                history_lines.append(f"{role_label}: {turn['content']}")
            history_str = '\n'.join(history_lines) + '\n'
        ret_mems_c = []
        for m_id in gt_ids:
            m_item = next((m for m in self.memory_bank if m['id'] == m_id), None)
            if m_item:
                actual_m = m_item['memory']
                t = actual_m.get('title') or actual_m.get('Title') or ''
                c = actual_m.get('caption') or actual_m.get('Caption') or ''
                m_res = f'记忆[{m_id}]: {t}'
                if c:
                    m_res += f' - {c}'
                ret_mems_c.append(m_res)
        mem_context = '检索到的记忆：\n' + '\n'.join(ret_mems_c) + '\n\n' if ret_mems_c else ''
        image_tags = ''.join([f'<image>\n' for _ in input_img_paths])
        v_action = 'mmu' if action in ['understand', 'pure_chat'] and input_img_paths else action
        if not input_img_paths and action != 'generate':
            v_action = 'pure_chat'
        if v_action == 'pure_chat':
            instruction_suffix = '请直接根据指令给出回答。使用 <think> 标签进行思考，使用 <ans> 标签输出结果。请务必全程使用中文进行思考和回答，严禁输出英文。'
        elif v_action == 'mmu':
            instruction_suffix = '给出详细的回答。使用 <think> 标签进行思考，使用 <ans> 标签输出结果。请务必全程使用中文进行思考和回答，严禁输出英文。'
        else:
            instruction_suffix = '给出详细的视觉描述（Prompt）。使用 <think> 标签进行思考，使用 <ans> 标签输出结果。请务必全程使用中文进行思考和回答，严禁输出英文。'
        if not input_img_paths:
            v_input = f'当前用户指令：{instr}\n任务类型：{v_action}\n{instruction_suffix}'
        else:
            v_input = f'{image_tags}\n{history_str}\n{mem_context}当前用户指令：{instr}\n任务类型：{v_action}\n请根据指令和参考图片进行思考，{instruction_suffix}'
        v_gt = self.understand_client.chat(v_input, images=input_img_paths if input_img_paths else None, temperature=0.3)
        if '<think>' not in v_gt and '<ans>' not in v_gt:
            v_gt = f'<think>\n（思考过程已略）\n</think>\n<ans>\n{v_gt}\n</ans>'
        elif '<think>' in v_gt and '<ans>' not in v_gt:
            parts = v_gt.split('</think>')
            if len(parts) > 1 and parts[1].strip():
                v_gt = parts[0] + '</think>\n<ans>\n' + parts[1].strip() + '\n</ans>'
        self.data_vlm_c.append({'images': input_img_paths, 'instruction': v_input, 'output': v_gt})
        print(f'    [Session {self.session_id}] VLM C Output (Final Sanitized):\n{v_gt}')
        self.conversation_log.append({'role': 'user', 'content': instr})
        clean_ans = re.sub('<(?:think|ans)>|</(?:think|ans)>', '', v_gt, flags=re.I).strip()
        if action in ['generate', 'edit', 'composite']:
            img_b = self.generate_client.generate(prompt=clean_ans, images=input_img_paths if input_img_paths else None)
            if img_b:
                p = self.save_image(img_b, turn_id)
                self.memory_bank.append(self.create_memory(p))
                self.conversation_log.append({'role': 'assistant', 'content': f'已生成图片：{p}'})
        else:
            self.conversation_log.append({'role': 'assistant', 'content': clean_ans})
        return True

    def save_all(self):
        global _shared_lock
        with _shared_lock:
            with open(os.path.join(self.data_dir, 'train_vlm_a.jsonl'), 'a') as f:
                for d in self.data_vlm_a:
                    f.write(json.dumps({'messages': [{'role': 'user', 'content': d['instruction']}, {'role': 'assistant', 'content': d['output']}], 'images': [d['image_path']]}, ensure_ascii=False) + '\n')
            with open(os.path.join(self.data_dir, 'train_vlm_b.jsonl'), 'a') as f:
                for d in self.data_vlm_b:
                    f.write(json.dumps({'messages': [{'role': 'user', 'content': d['instruction']}, {'role': 'assistant', 'content': d['output']}]}, ensure_ascii=False) + '\n')
            with open(os.path.join(self.data_dir, 'train_vlm_c.jsonl'), 'a') as f:
                for d in self.data_vlm_c:
                    f.write(json.dumps({'messages': [{'role': 'user', 'content': d['instruction']}, {'role': 'assistant', 'content': d['output']}], 'images': d['images']}, ensure_ascii=False) + '\n')
            try:
                sid_int = int(self.session_id)
                log_name = f'sess_{sid_int:04d}.json'
            except:
                log_name = f'sess_{self.session_id}.json'
            log_path = os.path.join(self.data_dir, log_name)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({'session_id': self.session_id, 'conversation': self.conversation_log, 'memory_bank': self.memory_bank}, f, ensure_ascii=False, indent=2)
            print(f'    [Session {self.session_id}] Session Log saved to: {log_path}')

def run_session(sid, data_dir, turns, lock):
    global _shared_lock
    _shared_lock = lock
    mgr = SessionManager(sid, data_dir)
    for t in range(turns):
        if not mgr.run_turn(t):
            break
    mgr.save_all()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sessions', type=int, default=5)
    parser.add_argument('--turns', type=int, default=4)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--output', type=str, default='./v2_with_negatives')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)
    manager = Manager()
    lock = manager.Lock()
    pool = Pool(processes=args.processes)
    from tqdm import tqdm
    results = []
    for i in range(args.num_sessions):
        res = pool.apply_async(run_session, args=(f'{i:04d}', args.output, args.turns, lock))
        results.append(res)
    pool.close()
    with tqdm(total=args.num_sessions, desc='Generating Sessions') as pbar:
        completed_count = 0
        while completed_count < args.num_sessions:
            new_completed = sum((1 for r in results if r.ready()))
            if new_completed > completed_count:
                pbar.update(new_completed - completed_count)
                completed_count = new_completed
            time.sleep(1)
    pool.join()
