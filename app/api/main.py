import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.api.schema import ChatRequest
from app.api.services import LegalApiService
from app.config import PROJECT_ROOT
from app.core.logger import get_logger, request_id_ctx, setup_app_logging

setup_app_logging(PROJECT_ROOT)
logger = get_logger(__name__)

app = FastAPI(
    title='Civil Law AI Agent',
    version='1.1.0',
    description='Hybrid Retrieval API with async ingestion',
)


@app.middleware('http')
async def context_logging_middleware(request: Request, call_next):
    request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
    token = request_id_ctx.set(request_id)
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        response.headers['X-Request-ID'] = request_id
        logger.info(f'{request.method} {request.url.path} | Status: {response.status_code} | Latency: {duration:.3f}s')
        return response
    finally:
        request_id_ctx.reset(token)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

service = LegalApiService()


@app.get('/')
async def root():
    return {'status': 'online', 'message': 'Law-judge API is running'}


@app.post('/api/chat')
async def chat(request: ChatRequest):
    try:
        result = service.chat(request.message, request.state_override)
        msgs = []
        for m in result.get('messages', []):
            if hasattr(m, 'type'):
                role = 'user' if m.type == 'human' else 'ai'
                content = getattr(m, 'content', '')
            elif isinstance(m, dict):
                role_raw = m.get('role') or m.get('type') or 'ai'
                role = 'user' if role_raw in {'human', 'user'} else 'ai'
                content = m.get('content', '')
            else:
                role = 'ai'
                content = str(m)
            msgs.append({'role': role, 'content': content})

        state_payload = dict(result)
        state_payload['messages'] = msgs
        return {
            'messages': msgs,
            'phase': result.get('phase', 'reception'),
            'intent': result.get('intent', 'unclear'),
            'state': state_payload,
        }
    except Exception as e:
        logger.error(f'Chat error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/chunks')
async def get_chunks():
    try:
        return service.get_all_chunks()
    except Exception as e:
        logger.error(f'Error fetching chunks: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/api/chunks/{collection_type}/{chunk_id}')
async def delete_chunk(collection_type: str, chunk_id: str):
    try:
        service.delete_chunk(collection_type, chunk_id)
        return {'status': 'success'}
    except Exception as e:
        logger.error(f'Error deleting chunk: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/api/files/{doc_type}/{file_name}')
async def delete_file(doc_type: str, file_name: str):
    try:
        if doc_type not in {'case', 'law', 'interpretation'}:
            raise HTTPException(status_code=400, detail='Invalid doc_type')
        from app.rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        success = retriever.delete_file_by_name(doc_type, file_name)
        if success:
            return {'status': 'success'}
        raise Exception('Backend deletion failed')
    except Exception as e:
        logger.error(f'Error deleting file {file_name}: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/ingestion/tasks')
async def get_ingestion_tasks():
    return service.get_ingestion_tasks()


@app.get('/api/metadata/registry')
async def get_metadata_registry():
    try:
        return service.retriever.get_available_metadata()
    except Exception as e:
        logger.error(f'Error fetching metadata registry: {e}')
        return {'elements': [], 'keywords': []}


@app.post('/api/upload')
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), doc_type: str | None = None):
    try:
        if doc_type and doc_type not in {'case', 'law', 'interpretation'}:
            raise HTTPException(status_code=400, detail='Invalid doc_type')

        temp_dir = Path('temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = temp_dir / file.filename

        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        service.ingestion_tasks[file.filename] = {'status': 'pending', 'error': None}
        logger.info(f'Accepted for background processing: {file.filename} as {doc_type}')
        background_tasks.add_task(service.ingest_file, file_path, doc_type)

        return {
            'status': 'success',
            'message': 'File accepted and processing in background.',
            'filename': file.filename,
        }
    except Exception as e:
        logger.error(f'Upload error: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app.api.main:app', host='0.0.0.0', port=8001, reload=False)
