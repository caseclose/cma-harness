import json
import os
import re
from pathlib import Path

def get_abs_path(path_str, base_dir):
    """转换相对路径为绝对路径，并修复重复目录名的问题"""
    if not path_str:
        return ''
    if os.path.isabs(path_str):
        return path_str
    if path_str.startswith('./'):
        path_str = path_str[2:]
    parent_dir = os.path.dirname(base_dir)
    dir_name = os.path.basename(base_dir)
    if path_str.startswith(dir_name):
        return os.path.abspath(os.path.join(parent_dir, path_str))
    else:
        return os.path.abspath(os.path.join(base_dir, path_str))

def reformat_vlm_a(input_dir, output_file):
    """转换 VLM A: 图像理解数据"""
    input_path = os.path.join(input_dir, 'train_vlm_a.jsonl')
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            abs_img = get_abs_path(data['image_path'], input_dir)
            swift_data = {'messages': [{'role': 'user', 'content': '<image>请详细分析这张图片，并以中文生成完整的摘要信息。输出必须是包含以下字段的 JSON 格式：title (标题), description (详细描述), caption (简短说明), tags (标签列表)。'}, {'role': 'assistant', 'content': json.dumps(data['ground_truth_memory'], ensure_ascii=False, indent=2)}], 'images': [abs_img]}
            f_out.write(json.dumps(swift_data, ensure_ascii=False) + '\n')
            count += 1
    print(f'VLM A 转换完成: {count} 条 -> {output_file}')

def reformat_vlm_b(input_dir, output_file):
    """转换 VLM B: 图像检索数据"""
    input_path = os.path.join(input_dir, 'train_vlm_b.jsonl')
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            mem_texts = []
            for mem in data['memory_bank']:
                summary = mem['summary']
                m_id = mem['id']
                mem_texts.append(f"记忆[{m_id}]: {summary.get('title', '')} - {summary.get('caption', '')}")
            mem_context = '\n'.join(mem_texts) if mem_texts else '（当前无历史记忆）'
            user_content = f"基于以下记忆库信息：\n{mem_context}\n\n用户查询：{data['user_query']}\n\n请根据用户查询和记忆库信息，选择最相关的图像ID列表。如果不需要引用历史图片，请输出空列表。"
            gt_ids = data['ground_truth_image_ids']
            assistant_content = f'图像ID: {gt_ids}'
            swift_data = {'messages': [{'role': 'user', 'content': user_content}, {'role': 'assistant', 'content': assistant_content}]}
            f_out.write(json.dumps(swift_data, ensure_ascii=False) + '\n')
            count += 1
    print(f'VLM B 转换完成: {count} 条 -> {output_file}')

def reformat_vlm_c(input_dir, output_file):
    """转换 VLM C: 指令翻译/MMU数据"""
    input_path = os.path.join(input_dir, 'train_vlm_c.jsonl')
    vlm_a_path = os.path.join(input_dir, 'train_vlm_a.jsonl')
    id_to_path = {}
    with open(vlm_a_path, 'r', encoding='utf-8') as f_a:
        for line in f_a:
            d = json.loads(line)
            key = f"{d['session_id']}_{d['id']}" if 'id' in d else f"{d['session_id']}_{d['turn_id']}"
            id_to_path[key] = get_abs_path(d['image_path'], input_dir)
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            ref_imgs = []
            for rid in data.get('ref_ids', []):
                k = f"{data['session_id']}_{rid}"
                if k in id_to_path:
                    ref_imgs.append(id_to_path[k])
            if not ref_imgs and data.get('user_uploaded_image_count', 0) > 0:
                pass
            image_tags = ''.join(['<image>'] * len(ref_imgs))
            history_list = data.get('history', [])
            history_str = ''
            if history_list:
                history_str = '对话历史：\n' + '\n'.join([f"{('用户' if h['role'] == 'user' else '助手')}: {h['content']}" for h in history_list]) + '\n\n'
            retrieved_mems = data.get('retrieved_memories', [])
            mem_texts = [f"记忆[{m['id']}]: {m['memory'].get('title', '')} - {m['memory'].get('caption', '')}" for m in retrieved_mems]
            mem_context = '检索到的记忆：\n' + '\n'.join(mem_texts) + '\n\n' if mem_texts else ''
            user_content = f"{image_tags}\n{history_str}{mem_context}当前用户指令：{data['query']}\n任务类型：{data['action']}\n请根据指令和参考图片进行思考，给出详细的回答或生成 Prompt。使用 <think> 标签进行思考，使用 <ans> 标签输出结果。"
            if data['action'] == 'mmu':
                assistant_content = data['gt_response']
            else:
                assistant_content = data['gt_prompt']
            swift_data = {'messages': [{'role': 'user', 'content': user_content}, {'role': 'assistant', 'content': assistant_content}], 'images': ref_imgs}
            f_out.write(json.dumps(swift_data, ensure_ascii=False) + '\n')
            count += 1
    print(f'VLM C 转换完成: {count} 条 -> {output_file}')
if __name__ == '__main__':
    DATA_DIR = './data/simulated_data_0114_1'
    OUTPUT_DIR = './data/converted_data_0114_1'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reformat_vlm_a(DATA_DIR, os.path.join(OUTPUT_DIR, 'train_vlm_a_swift.jsonl'))
    reformat_vlm_b(DATA_DIR, os.path.join(OUTPUT_DIR, 'train_vlm_b_swift.jsonl'))
    reformat_vlm_c(DATA_DIR, os.path.join(OUTPUT_DIR, 'train_vlm_c_swift.jsonl'))
