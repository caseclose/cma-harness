"""
多模态对话Agent API服务
提供HTTP接口用于：
1. 创建新session
2. 上传图片并生成摘要
3. 获取memory bank
4. 清空memory bank
"""
import os
import json
import time
from typing import Optional, List
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from multimodal_agent_service import MultimodalAgentService, MemoryBank
_service_start_time = time.time()
DEFAULT_GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
os.environ['CUDA_VISIBLE_DEVICES'] = DEFAULT_GPUS
os.environ['MAX_PIXELS'] = '100352000'
app = FastAPI(title='Multimodal Agent Service', description='多模态对话Agent服务API', version='1.0.0')
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['*'])
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
service: Optional[MultimodalAgentService] = None
UPLOAD_DIR = Path('/tmp/agent_uploads')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event('startup')
async def startup_event():
    """服务启动时初始化"""
    global service
    print('Initializing Multimodal Agent Service...')
    print(f'Available GPUs: {DEFAULT_GPUS}')
    vlm_a_gpus = os.environ.get('VLM_A_GPUS')
    if vlm_a_gpus:
        vlm_a_gpus = [int(x.strip()) for x in vlm_a_gpus.split(',')]
    else:
        vlm_a_gpus = None
    vlm_b_gpus = os.environ.get('VLM_B_GPUS')
    if vlm_b_gpus:
        vlm_b_gpus = [int(x.strip()) for x in vlm_b_gpus.split(',')]
    else:
        vlm_b_gpus = None
    service = MultimodalAgentService(vlm_a_gpus=vlm_a_gpus, vlm_b_gpus=vlm_b_gpus)
    print('Service initialized successfully!')

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message: str

class ImageSummaryRequest(BaseModel):
    session_id: str
    image_path: Optional[str] = None

class ImageSummaryResponse(BaseModel):
    memory_id: int
    memory: dict
    turn_count: int
    message: str

class MemoryBankResponse(BaseModel):
    memories: list
    count: int

class MessageResponse(BaseModel):
    message: str

class ImageGenerationRequest(BaseModel):
    user_instruction: str
    action_type: str = 'edit'
    reference_image_ids: Optional[List[int]] = None
    use_vlm_b_retrieval: bool = True

class ImageGenerationResponse(BaseModel):
    image_url: str
    image_path: str
    generation_prompt: str
    retrieved_image_ids: List[int]
    memory_id: int
    memory: dict
    turn_count: int
    message: str

@app.post('/api/v1/sessions', response_model=SessionResponse)
async def create_session(session_id: Optional[str]=None):
    """
    创建新的session
    
    Args:
        session_id: 可选的session ID，如果不提供则自动生成
    
    Returns:
        Session信息
    """
    try:
        new_session_id = service.create_session(session_id)
        session_info = service.get_session(new_session_id)
        return SessionResponse(session_id=new_session_id, created_at=session_info['created_at'], message='Session created successfully')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/v1/sessions/{session_id}/images/upload', response_model=ImageSummaryResponse)
async def upload_and_summarize_image(session_id: str, file: UploadFile=File(...)):
    """
    上传图片并生成摘要
    
    Args:
        session_id: session ID
        file: 上传的图片文件
    
    Returns:
        摘要结果
    """
    try:
        if session_id not in service.sessions:
            raise HTTPException(status_code=404, detail=f'Session {session_id} not found')
        file_extension = Path(file.filename).suffix or '.jpg'
        saved_path = UPLOAD_DIR / f'{session_id}_{file.filename}'
        with open(saved_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        result = service.summarize_image(session_id, str(saved_path))
        return ImageSummaryResponse(memory_id=result['memory_id'], memory=result['memory'], turn_count=result['turn_count'], message='Image summarized successfully')
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/v1/sessions/{session_id}/images/summarize', response_model=ImageSummaryResponse)
async def summarize_existing_image(session_id: str, image_path: str=Form(...)):
    """
    对已有图片路径进行摘要
    
    Args:
        session_id: session ID
        image_path: 图片路径（服务器上的绝对路径）
    
    Returns:
        摘要结果
    """
    try:
        if session_id not in service.sessions:
            raise HTTPException(status_code=404, detail=f'Session {session_id} not found')
        result = service.summarize_image(session_id, image_path)
        return ImageSummaryResponse(memory_id=result['memory_id'], memory=result['memory'], turn_count=result['turn_count'], message='Image summarized successfully')
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/v1/sessions/{session_id}/memory-bank', response_model=MemoryBankResponse)
async def get_memory_bank(session_id: str):
    """
    获取session的memory bank
    
    Args:
        session_id: session ID
    
    Returns:
        Memory bank信息
    """
    try:
        memory_bank = service.get_memory_bank(session_id)
        return MemoryBankResponse(memories=memory_bank['memories'], count=memory_bank['count'])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/v1/sessions/{session_id}/memory-bank', response_model=MessageResponse)
async def clear_memory_bank(session_id: str):
    """
    清空session的memory bank
    
    Args:
        session_id: session ID
    
    Returns:
        操作结果
    """
    try:
        service.clear_memory_bank(session_id)
        return MessageResponse(message='Memory bank cleared successfully')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/v1/sessions/{session_id}')
async def get_session_info(session_id: str):
    """
    获取session信息
    
    Args:
        session_id: session ID
    
    Returns:
        Session信息
    """
    try:
        session = service.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f'Session {session_id} not found')
        return {'session_id': session['session_id'], 'created_at': session['created_at'], 'turn_count': session['turn_count'], 'memory_count': len(session['memory_bank'].memories)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/v1/sessions/{session_id}/images/generate', response_model=ImageGenerationResponse)
async def generate_image(session_id: str, request: ImageGenerationRequest):
    """
    生成图片的完整流程：
    1. 调用VLM C生成图像生成提示词
    2. 调用Gemini API生成图片
    3. 保存生成的图片
    4. 调用VLM A对生成的图片进行摘要
    5. 存入memory bank
    
    Args:
        session_id: session ID
        request: 图片生成请求（包含用户指令、操作类型、参考图像ID等）
    
    Returns:
        生成结果（包含图片URL、路径、摘要信息等）
    """
    try:
        if session_id not in service.sessions:
            raise HTTPException(status_code=404, detail=f'Session {session_id} not found')
        result = service.generate_image(session_id=session_id, user_instruction=request.user_instruction, action_type=request.action_type, reference_image_ids=request.reference_image_ids, use_vlm_b_retrieval=request.use_vlm_b_retrieval)
        return ImageGenerationResponse(image_url=result['image_url'], image_path=result['image_path'], generation_prompt=result['generation_prompt'], retrieved_image_ids=result.get('retrieved_image_ids', []), memory_id=result['memory_id'], memory=result['memory'], turn_count=result['turn_count'], message='Image generated and summarized successfully')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health_check():
    """健康检查"""
    return {'status': 'healthy', 'service': 'Multimodal Agent Service'}

@app.get('/metrics')
async def metrics():
    """Prometheus格式的指标端点"""
    uptime = time.time() - _service_start_time
    metrics_text = f'# HELP service_uptime_seconds Service uptime in seconds\n# TYPE service_uptime_seconds gauge\nservice_uptime_seconds {uptime:.2f}\n\n# HELP service_info Service information\n# TYPE service_info gauge\nservice_info{{service="multimodal_agent"}} 1\n'
    return Response(content=metrics_text, media_type='text/plain; version=0.0.4')
static_dir = Path(__file__).parent / 'static'
if static_dir.exists():
    app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')

    @app.get('/')
    async def index():
        """返回Web界面"""
        index_file = static_dir / 'index.html'
        if index_file.exists():
            return FileResponse(str(index_file))
        return {'message': 'Web interface not found. Please check static/index.html'}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multimodal Agent API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    uvicorn.run('agent_api_server:app', host=args.host, port=args.port, reload=args.reload)
