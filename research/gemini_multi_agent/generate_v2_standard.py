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
        user_prompt = f"""你是一个真实的人类用户，正在与AI绘图助手聊天。\n当前任务：{chosen_action}\n\n历史记录中的图片：\n{memory_summary}\n\n【核心指令】：\n请根据任务类型说一句话。\n1. **严禁**在回复中使用“ID”、“程序ID”或任何数字编号来指代图片（除非是“第一张”、“第二张”这种自然口语）。\n2. **指代方式**：\n   - 使用视觉特征：如“那个有月球表面的图”、“穿红衣服的人”。\n   - 使用相对位置：如“最后那张”、“第一张”、“倒数第二张”、“刚才生成的那个”。\n   - 模糊指代：如“这图里”、“之前那张”。\n3. **说话语气**：极其自然、随意，像是在和朋友聊天。\n\n请输出 JSON：\n{{\n    "instruction": "你的自然语言指令（严禁出现'ID'字样）",\n    "ref_image_ids": [你引用的程序ID列表（从上面提示中获取），不引用则为 []],\n    "should_upload_new_image": bool\n}}\n"""
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
        image_task_examples = '\n=== 范例展示 (绘画、编辑与融合任务) ===\n示例 1：全新的文生图 (generate)\n用户指令：帮我画一张赛博朋克风格的街道，要有雨后的质感。\n助手：\n<think>\n1. **任务类型**：生成新图像（generate）。\n2. **核心元素提取**：赛博朋克、未来都市、街道、下雨。\n3. **视觉描述逻辑**：重点描述霓虹灯在湿滑地面的反射、空气中的水雾、高饱和度的冷色调（青/紫）。\n</think>\n<ans>\n视觉描述：一张赛博朋克风格的未来都市街道图。大雨刚过，湿漉漉的柏油马路清晰地倒映着两旁绚丽的霓虹灯招牌。空气中弥漫着蓝色的水雾，高耸的建筑带有密集的金属质感细节，整体色调以电光蓝和洋红色为主，营造出一种迷幻且先进的氛围。\n</ans>\n\n示例 2：基于单张图的编辑 (edit)\n用户指令：刚才那张图里的天空太暗了，能不能换成那种火烧云的傍晚感觉？\n助手：\n<think>\n1. **任务类型**：图片编辑（edit）。\n2. **参考对象**：当前传给你的第一张图（Image 1）。\n3. **修改策略**：保持 Image 1 中的街道、建筑和行人等主体不变，仅针对背景天空进行替换。\n4. **一致性维护**：地面的倒影也应根据火烧云的橙红色光影进行同步偏移。\n</think>\n<ans>\n视觉描述：请参考 Image 1，将其背景天空替换为绚烂的火烧云景象。原本暗淡的天空现在布满了橙红色和金黄色的云层，温暖的暮色洒在 Image 1 的城市建筑轮廓线上。保持 Image 1 的街道和行人不变，但地面积水中应映照出与火烧云同步的瑰丽橙红色反光。\n</ans>\n\n示例 3：基于多张图的编辑/多图元素的融合 (edit/composite)\n用户指令：把第一张图里的那只赛博朋克猫，放到刚才那张江南水乡的背景里，弄个穿越的效果。\n助手：\n<think>\n1. **任务类型**：基于多张图的编辑/多图元素的融合（edit/composite）。\n2. **参考对象**：当前传给你的第一张图（Image 1）和第二张图（Image 2）。\n2. **元素提取**：从 Image 1 提取主体（戴墨镜的赛博朋克猫），从 Image 2 提取背景（江南水乡古镇）。\n3. **融合逻辑**：将 Image 1 中高科技、冷色调的猫作为前景，完美嵌入到 Image 2 那古朴、暖色调的水乡建筑背景中。\n4. **细节处理**：猫身上的发光背心应在 Image 2 的石板路上产生微弱的电子光影。\n</think>\n<ans>\n视觉描述：一张跨时空融合图像。请提取 Image 1 中的核心主体（戴墨镜、穿发光战术背心的黑猫），并将其放置在 Image 2 的背景中（白墙黛瓦、石拱桥和晨雾缭绕的水乡古镇）。确保黑猫的姿态与 Image 2 的环境光影融合自然，猫身上的电子荧光与 Image 2 古朴的水墨色调形成强烈反差，营造出未来科技入侵传统世界的奇妙感觉。\n</ans>\n==========================\n'
        text_task_examples = '\n=== 范例展示 (理解与对话任务) ===\n示例 1：视觉理解与分析 (mmu)\n用户指令：这张图里的人是在看星星吗？\n助手：\n<think>\n1. **任务类型**：视觉理解（mmu）。\n2. **画面细节扫描**：观察到落地窗前有一个仰望天空的黑色剪影，双手举在眼前。\n3. **逻辑推理**：仰头姿态配合璀璨星空背景，手部动作疑似持望远镜，判定为观星行为。\n</think>\n<ans>\n是的，从人物仰望星空的姿态和手持望远镜的动作来看，他确实正在全神贯注地观测星空。\n</ans>\n\n示例 2：纯文本对话与建议 (pure_chat)\n用户指令：哈尔滨冬天去玩冷吗？有什么推荐？\n助手：\n<think>\n1. **任务类型**：纯文本对话（pure_chat）。\n2. **知识检索**：哈尔滨冬季气候极度寒冷，通常在-20℃以下。\n3. **回复策略**：确认寒冷程度，并给出简短的景点建议（冰雪大世界）和保暖建议。\n</think>\n<ans>\n哈尔滨的冬天确实非常寒冷，气温经常掉到零下20多度。去玩的话建议一定要去冰雪大世界看看，但务必准备好加厚的羽绒服、保暖内衣和防滑雪地靴哦！\n</ans>\n==========================\n'
        if v_action in ['generate', 'edit', 'composite']:
            chosen_example = image_task_examples
        else:
            chosen_example = text_task_examples
        if not input_img_paths:
            v_input = f'{chosen_example}\n当前用户指令：{instr}\n任务类型：{v_action}\n{instruction_suffix}'
        else:
            v_input = f'{chosen_example}\n{image_tags}\n{history_str}\n{mem_context}当前用户指令：{instr}\n任务类型：{v_action}\n请根据指令和参考图片进行思考，{instruction_suffix}'
        v_gt = self.understand_client.chat(v_input, images=input_img_paths if input_img_paths else None, temperature=0.3)
        if '<think>' not in v_gt and '<ans>' not in v_gt:
            v_gt = f'<think>\n正在根据当前指令和历史信息进行综合分析...\n</think>\n<ans>\n{v_gt}\n</ans>'
        elif '<think>' in v_gt and '<ans>' not in v_gt:
            parts = v_gt.split('</think>')
            if len(parts) > 1 and parts[1].strip():
                v_gt = parts[0] + '</think>\n<ans>\n' + parts[1].strip() + '\n</ans>'
        self.data_vlm_c.append({'images': input_img_paths, 'instruction': v_input, 'output': v_gt})
        print(f'    [Session {self.session_id}] VLM C Output (Final Sanitized):\n{v_gt}')
        self.conversation_log.append({'role': 'user', 'content': instr})
        ans_match = re.search('<ans>(.*?)</ans>', v_gt, re.S | re.I)
        if ans_match:
            clean_ans = ans_match.group(1).strip()
        else:
            clean_ans = re.sub('<(?:think|ans)>|</(?:think|ans)>', '', v_gt, flags=re.I).strip()
        if action in ['generate', 'edit', 'composite']:
            print(f'    [Session {self.session_id}] Calling Image Generator...')
            img_b = self.generate_client.generate(prompt=clean_ans, images=input_img_paths if input_img_paths else None)
            if img_b:
                p = self.save_image(img_b, turn_id)
                self.memory_bank.append(self.create_memory(p))
                self.conversation_log.append({'role': 'assistant', 'content': f'已生成图片：{p}'})
                print(f'    [Session {self.session_id}] Image saved: {p}')
            else:
                self.conversation_log.append({'role': 'assistant', 'content': '抱歉，绘画失败了。'})
                print(f'    [Session {self.session_id}] ⚠️ Image generation FAILED after retries.')
        else:
            self.conversation_log.append({'role': 'assistant', 'content': clean_ans})
            print(f'    [Session {self.session_id}] Task {action} completed.')
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
    parser.add_argument('--output', type=str, default='./v2_standard')
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
