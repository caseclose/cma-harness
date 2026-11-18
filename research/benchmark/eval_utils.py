import json
from cma_api_client import GeminiJudge
from typing import List, Optional, Dict
from PIL import Image

def evaluate_mmu(judge: GeminiJudge, user_input: str, agent_response: str, context_images: List[str]=None):
    """MMU task scoring using a 1-5 scale."""
    prompt = f'You are a professional multimodal evaluation specialist.\n\n**Task: UNDERSTAND (Visual Reasoning/MMU)**\n**Evaluation Scale (1 to 5):**\n1. **factuality_score**: Does the answer accurately reflect visual facts in the image(s)? (5: Perfect, 1: Severe Hallucinations)\n2. **reference_score**: Does it correctly resolve references like "it", "the previous one", or specific objects?\n3. **reasoning_score**: Is the logic sound and does it capture key details for complex questions?\n\n**User Input:**\n{user_input}\n\n**Agent Response:**\n{agent_response}\n\n**Output Requirement:**\nReturn ONLY a JSON object:\n{{\n  "factuality_score": X,\n  "reference_score": X,\n  "reasoning_score": X,\n  "total_score": X,\n  "reasoning": "Brief explanation for each score."\n}}'
    res_text = judge.chat(prompt, images=context_images)
    try:
        res_text = res_text.strip().replace('```json', '').replace('```', '')
        return json.loads(res_text)
    except:
        return {'error': 'Failed to parse judge response', 'raw': res_text}

def evaluate_edit(judge: GeminiJudge, instruction: str, source_image: str, generated_image: str):
    """Specialized scoring for EDIT tasks based on image editing criteria (1-5 scale)."""
    prompt = f'You are a professional digital artist and image evaluation specialist.\nCompare the **Original Image** and the **Edited Image** based on the **Instruction**.\n\n**Editing Instruction:** {instruction}\n\n**Evaluation Scale (1 to 5):**\n1. **consistency_score**: Evaluate visual consistency focusing exclusively on elements NOT specified for change. (5: Perfect, 1: Severe Inconsistency)\n2. **instruction_score**: Evaluate how faithfully the edited image fulfills the instruction. (5: Perfect Compliance, 1: Non-Compliance)\n3. **quality_score**: Evaluate perceptual quality, structural coherence, and absence of artifacts. (5: Excellent, 1: Severe Errors/Artifacts)\n\n**Reasoning Steps:**\n1. **Detect Difference**: What visually changed?\n2. **Expected Visual Caption**: How should it look if perfectly followed?\n3. **Instruction Match**: Compare differences to expectations.\n\n**Output Requirement:**\nReturn ONLY a JSON object:\n{{\n  "consistency_score": X,\n  "instruction_score": X,\n  "quality_score": X,\n  "total_score": X,\n  "reasoning": "1. Detect Difference... 2. Expected Visual Caption... 3. Instruction Match... 4. Decision"\n}}'
    res_text = judge.chat(prompt, images=[source_image, generated_image])
    try:
        res_text = res_text.strip().replace('```json', '').replace('```', '')
        return json.loads(res_text)
    except:
        return {'error': 'Failed to parse judge response', 'raw': res_text}

def evaluate_generate(judge: GeminiJudge, instruction: str, generated_image: str, history_images: List[str]=None):
    """Specialized scoring for GENERATE tasks (1-5 scale)."""
    prompt = f"""You are a professional image generation evaluation specialist.\n\n**Task: GENERATE (Text-to-Image)**\n**Instruction:** {instruction}\n\n**Evaluation Scale (1 to 5):**\n1. **instruction_score**: Does the image match the user's intent and include all requested elements?\n2. **consistency_score**: (If history images provided) Does it maintain style/identity consistency?\n3. **quality_score**: Overall perceptual quality, naturalness, and absence of artifacts.\n\n**Output Requirement:**\nReturn ONLY a JSON object:\n{{\n  "instruction_score": X,\n  "consistency_score": X,\n  "quality_score": X,\n  "total_score": X,\n  "reasoning": "Detailed breakdown of the generation quality and compliance."\n}}"""
    images = []
    if history_images:
        images.extend(history_images)
    images.append(generated_image)
    res_text = judge.chat(prompt, images=images)
    try:
        res_text = res_text.strip().replace('```json', '').replace('```', '')
        return json.loads(res_text)
    except:
        return {'error': 'Failed to parse judge response', 'raw': res_text}

def evaluate_chat(judge: GeminiJudge, user_input: str, agent_response: str, chat_history: str=None):
    """Text-only chat task scoring (1-5 scale)."""
    prompt = f"""You are a professional AI conversation evaluation specialist.\n\n**Task: CHAT (Text-only)**\n**Evaluation Scale (1 to 5):**\n1. **instruction_score**: Did the agent follow the user's intent and formatting?\n2. **context_score**: Is the response coherent with the previous chat history?\n3. **quality_score**: Is the information accurate, helpful, and logically clear?\n\n**Chat History:**\n{(chat_history if chat_history else 'None')}\n\n**User Input:** {user_input}\n**Agent Response:** {agent_response}\n\n**Output Requirement:**\nReturn ONLY a JSON object:\n{{\n  "instruction_score": X,\n  "context_score": X,\n  "quality_score": X,\n  "total_score": X,\n  "reasoning": "Evaluation of the dialogue quality."\n}}"""
    res_text = judge.chat(prompt)
    try:
        res_text = res_text.strip().replace('```json', '').replace('```', '')
        return json.loads(res_text)
    except:
        return {'error': 'Failed to parse judge response', 'raw': res_text}
