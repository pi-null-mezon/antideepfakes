# -----------------------------------------------------------------
# Http server for the face liveness detection
#
# (C) 2021 Alex A. Taranov, Moscow, Russia, taransanya@pi-mezon.ru
# -----------------------------------------------------------------

import os
import json
import uuid

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

import torch
from deepfakes import DD256x60x01, DD224x90x02, liveness_score
from face import YuNetFaceDetector
from landmarks import LWADetector

path = os.getenv('PATH_PREFIX', '')
prefix = f"/v1/{path}/liveness" if path != '' else "/v1/liveness"

app = FastAPI(
    title="SystemFailure's Deepfake Detection HTTP service",
    description="Сервис проверки подлинности видеозаписей лица",
    version="1.0.0",
    openapi_tags=[{"name": "ЕБС"}]
)

EBS = {
    "content type": {"http": 400, "code": "LDE-002001", "message": "Неверный Content-Type HTTP-запроса"},
    "method type": {"http": 400, "code": "LDE-002002", "message": "Неверный метод HTTP-запроса"},
    "mnemonic": {"http": 400, "code": "LDE-002003", "message": "Неверный формат метаданных"},
    "face not found": {"http": 400, "code": "LDE-002006", "message": "Лицо не найдено"},
    "too small face": {"http": 400, "code": "LDE-002008", "message": "Лицо слишком маленького размера"},
    "multiple faces": {"http": 400, "code": "LDE-002007", "message": "Найдено больше одного лица"},
    "sample read": {"http": 400, "code": "LDE-002004", "message": "Не удалось прочитать биометрический образец"},
    "internal": {"http": 500, "code": "LDE-001001", "message": "Внутренняя ошибка БП обнаружения витальности"},
    "content": {"http": 400, "code": "LDE-002005", "message": "Неверная multiparted-часть HTTP запроса"},
}

class HealthModel(BaseModel):
    status: int = Field(default=0, description="0 - исправен, > 0 - неисправен")
    message: Optional[str] = Field(description="справочная информация", default='')


class ErrorModel(BaseModel):
    code: str = Field(description="код ошибки в соответствии с методическими рекомендациями ЕБС")
    error: str = Field(description="расшифровка кода ошибки")


class LivenessModel(BaseModel):
    passed: bool = Field(description="решение по подлинности биометрического предъявления")
    score: float = Field(description="степень уверенности в подлинности биометрического предъявления "
                                     "от 0.0 до 1.0 (чем выше, тем выше уверенность)")

face_detector = None
landmarks_detector = None
deepfake_detectors = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event('startup')
async def startup_event():
    print(f"  - inference backend: '{device}'")
    global face_detector
    face_detector = YuNetFaceDetector("./weights/final/fd.onnx")
    global landmarks_detector
    landmarks_detector = LWADetector("./weights/final/ld.onnx")
    global deepfake_detectors
    deepfake_detectors = [
        DD256x60x01([
            './weights/final/256x60x0.1/tmp_dd_on_effnet_v2_s@256x60x0.1.jit',
        ], device),
    ]
    if os.getenv('DOUBLE_CHECK', False):
        deepfake_detectors.append([
            DD224x90x02([
                './weights/final/224x90x0.2/tmp_dd_on_effnet_v2_s@224x90x0.2.jit',
            ], device)
        ])
    print("  - warming up nets, please wait...")
    for i in range(5):  # several iteration is needed to optimize CUDA calls
        liveness_score('./resources/fake.mp4', deepfake_detectors, face_detector, landmarks_detector, delete_file=False)


@app.get(f"{prefix}/health",
         response_class=JSONResponse,
         tags=["ЕБС"],
         response_model=HealthModel,
         responses={500: {"model": ErrorModel}},
         summary="провести автодиагностику")
async def get_status():
    live = liveness_score("./resources/live.mp4",
                          deepfake_detectors, face_detector, landmarks_detector, delete_file=False)
    print(f"live sample liveness score: {live:.3f}")
    fake = liveness_score("./resources/fake.mp4",
                          deepfake_detectors, face_detector, landmarks_detector, delete_file=False)
    print(f"deepfake sample liveness score: {fake:.3f}")
    if fake > live:
        return {"status": 1, "message": "Биопроцессор неисправен!"}
    return {"status": 0, "message": "Биопроцессор исправен"}


@app.post(f"{prefix}/detect",
          response_class=JSONResponse,
          tags=["ЕБС"],
          response_model=LivenessModel,
          responses={400: {"model": ErrorModel}},
          summary="проверить видео")
async def process(request: Request,
                  bio_sample: UploadFile = File(None, description="файл для проверки [mp4, webm]"),
                  metadata: UploadFile = File(None, description="метаданные [json]")):
    if 'multipart/form-data' not in request.headers['content-type']:
        print("Wrong content-type", flush=True)
        error = 'content type'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
        # metadata parsing
    if 'application/json' not in metadata.content_type:
        print("Wrong metadata content-type'", flush=True)
        error = 'content'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    metadata = await metadata.read()
    json_metadata = json.loads(metadata)
    if "mnemonic" not in json_metadata:
        print("metadata does not contain 'mnemonic'", flush=True)
        error = 'mnemonic'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    if json_metadata["mnemonic"] != "passive-instructions":
        print("Only 'passive-instructions' mnemonic allowed", flush=True)
        error = 'content'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    if "actions" not in json_metadata:
        print("Metadata does not contain 'actions'", flush=True)
        error = 'mnemonic'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    if len(json_metadata["actions"]) != 1:
        print("Length of 'actions' more than one", flush=True)
        error = 'content'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    if "type" not in json_metadata["actions"][0]:
        print("'actions' does not contain type field", flush=True)
        error = 'mnemonic'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    if json_metadata["actions"][0]["type"] != "deepfake-type":
        print("'actions' does not contain 'deepfake-type' items", flush=True)
        error = 'content'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    # bio_sample processing
    if bio_sample.content_type not in ['video/mp4', 'video/webm']:
        print("Wrong metadata content-type'", flush=True)
        error = 'content'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    bio_sample_bytes = await bio_sample.read()
    if len(bio_sample_bytes) == 0:
        print("Zero size bio_sample", flush=True)
        error = 'sample read'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    video_filename = f"./weights/{uuid.uuid4().hex}.{bio_sample.content_type.rsplit('/', 1)[1]}"
    with open(video_filename, 'wb') as o_f:
        o_f.write(bio_sample_bytes)

    try:
        score = liveness_score(video_filename, deepfake_detectors, face_detector, landmarks_detector, delete_file=True)
    except IOError as ex:
        print("Can not decode bio_sample", flush=True)
        error = 'sample read'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    #if len(list_of_landmarks) > 1:
    #    error = 'multiple faces'
    #    return JSONResponse(status_code=EBS[error]['http'],
    #                        content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    if score is None:
        error = 'face not found'
        return JSONResponse(status_code=EBS[error]['http'],
                            content={"code": EBS[error]['code'], "message": EBS[error]['message']})
    return {"score": score, "passed": bool(score > 0.5)}


@app.get(f"{prefix}/detect",
         response_class=JSONResponse,
         tags=["ЕБС"],
         response_model=ErrorModel,
         responses={400: {"model": ErrorModel}},
         summary="затычка для API ЕБС",
         include_in_schema=False)
async def dummy_process():
    print("Not allowed HTTP method", flush=True)
    error = 'method type'
    return JSONResponse(status_code=EBS[error]['http'],
                        content={"code": EBS[error]['code'], "message": EBS[error]['message']})


if __name__ == '__main__':
    server_name = 'SystemFailure©: Deepfake Detection HTTP service'
    print('=' * len(server_name), flush=True)
    print(server_name, flush=True)
    print('=' * len(server_name), flush=True)
    print(f'Version: {app.version}', flush=True)
    print('API: EBS', flush=True)
    print('Release date: 12.11.2022', flush=True)
    print('=' * len(server_name), flush=True)
    print('Configuration:', flush=True)
    http_srv_addr = os.getenv("APP_ADDR", "0.0.0.0")
    http_srv_port = int(os.getenv("APP_PORT", 5000))
    workers = int(os.getenv("WORKERS", 1))
    print(f"  - prefix: '{prefix}'", flush=True)
    print(f"  - workers: {workers}", flush=True)
    print(f"  - double check: {os.getenv('DOUBLE_CHECK', False)}", flush=True)
    uvicorn.run('httpsrv:app', host=http_srv_addr, port=http_srv_port, workers=workers)
