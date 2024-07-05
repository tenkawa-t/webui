from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import cv2
import numpy as np
import base64
import torch
import torchvision
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import os
import traceback
from ultralytics import YOLO, FastSAM

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# モデル定義
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# モデルのロード関数
def load_yolov5_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def load_yolox_model():
    return torch.hub.load('Megvii-BaseDetection/YOLOX', 'yolox_s', pretrained=True)

def load_fastsam_model():
    return FastSAM('FastSAM-x.pt')  # モデルのパスを適切に設定してください

def load_yolov8_model():
    return YOLO(model='yolov8x.pt')  # モデルのパスを適切に設定してください

def load_resnet_model():
    return resnet50(pretrained=True)

def load_autoencoder_model():
    model = AutoEncoder()
    # 事前学習済みの重みがある場合はここでロード
    # model.load_state_dict(torch.load('autoencoder_weights.pth'))
    return model

# 画像処理関数
def decode_image(image_data):
    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def apply_operation(image, operation, params):
    if operation == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif operation == 'blur':
        return cv2.GaussianBlur(image, (params['ksize'], params['ksize']), params['sigmaX'])
    elif operation == 'edge_detection':
        return cv2.Canny(image, params['threshold1'], params['threshold2'])
    elif operation == 'threshold':
        _, result = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), params['thresh'], params['maxval'], params['type'])
        return result
    elif operation == 'yolov5':
        return yolov5_detect(image)
    elif operation == 'yolox':
        return yolox_detect(image)
    elif operation == 'fastsam':
        return fastsam_segment(image)
    elif operation == 'yolov8':
        return yolov8_detect(image)
    elif operation == 'resnet':
        return resnet_classify(image)
    elif operation == 'autoencoder':
        return autoencoder_anomaly_detect(image)
    else:
        raise ValueError(f"Unknown operation: {operation}")

def yolov5_detect(image):
    model = load_yolov5_model()
    results = model(image)
    return results.render()[0]

def yolox_detect(image):
    model = load_yolox_model()
    results = model(image)
    return draw_results(image, results)

def fastsam_segment(image):
    model = load_fastsam_model()
    results = model(image, device='cpu')
    return results[0].plot()

def yolov8_detect(image):
    model = load_yolov8_model()
    results: list = model.predict(image, save=False)
    return results[0].plot()

def resnet_classify(image):
    model = load_resnet_model()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    probabilities = F.softmax(output[0], dim=0)
    return draw_classification_results(image, probabilities)

def autoencoder_anomaly_detect(image):
    model = load_autoencoder_model()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(tensor)
    mse_loss = F.mse_loss(tensor, reconstructed, reduction='none').mean(dim=[1,2,3])
    return draw_anomaly_results(image, mse_loss.item())

# 結果描画関数
def draw_results(image, results):
    # 検出結果を画像上に描画
    return image

def draw_classification_results(image, probabilities):
    # 分類結果を描画するコードを実装
    return image

def draw_anomaly_results(image, anomaly_score):
    # 異常検知の結果を描画するコードを実装
    return image

# コードスニペット生成関数
def get_code_snippet(index, operation, params):
    comment = f"# Step {index}: {operation.capitalize()}"
    if operation == 'grayscale':
        code = "image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)"
    elif operation == 'blur':
        code = f"image = cv2.GaussianBlur(image, ({params['ksize']}, {params['ksize']}), {params['sigmaX']})"
    elif operation == 'edge_detection':
        code = f"image = cv2.Canny(image, {params['threshold1']}, {params['threshold2']})"
    elif operation == 'threshold':
        code = f"_, image = cv2.threshold(image, {params['thresh']}, {params['maxval']}, {params['type']})"
    elif operation == 'yolov5':
        code = "results = yolov5_model(image)\nimage = results.render()[0]"
    elif operation == 'yolox':
        code = "results = yolox_model(image)\nimage = draw_results(image, results)"
    elif operation == 'fastsam':
        code = "results = fastsam_model(image)\nimage = results[0].plot()"
    elif operation == 'yolov8':
        code = "results = yolov8_model(image)\nimage = results[0].plot()"
    elif operation == 'resnet':
        code = "output = resnet_model(torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0))\nimage = draw_classification_results(image, F.softmax(output[0], dim=0))"
    elif operation == 'autoencoder':
        code = "reconstructed = autoencoder_model(torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0))\nmse_loss = F.mse_loss(tensor, reconstructed, reduction='none').mean(dim=[1,2,3])\nimage = draw_anomaly_results(image, mse_loss.item())"
    else:
        code = f"# Unknown operation: {operation}"
    
    return f"{comment}\n{code}"

# FastAPI ルート
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_image(request: dict):
    try:
        image_data = request['image']
        pipeline = request['pipeline']

        image = decode_image(image_data)
        code_snippets = []
        
        for index, step in enumerate(pipeline, start=1):
            operation = step['operation']
            params = step.get('params', {})
            try:
                print(f"Processing step: {operation}")  # デバッグ情報
                image = apply_operation(image, operation, params)
                code_snippets.append(get_code_snippet(index, operation, params))
            except Exception as e:
                print(f"Error in operation '{operation}': {str(e)}")  # デバッグ情報
                raise ValueError(f"Error in operation '{operation}': {str(e)}")

        result_image = encode_image(image)
        code = "\n\n".join(code_snippets)
        return JSONResponse(content={'image': result_image, 'code': code})
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        error_traceback = traceback.format_exc()
        print(f"Error: {error_message}")
        print(f"Traceback: {error_traceback}")
        raise HTTPException(status_code=500, detail=error_message)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_message = f"An unexpected error occurred: {str(exc)}"
    error_traceback = traceback.format_exc()
    print(f"Error: {error_message}")
    print(f"Traceback: {error_traceback}")
    return JSONResponse(
        status_code=500,
        content={"message": error_message, "traceback": error_traceback},
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("opencvwebapply_uifix:app", host="127.0.0.1", port=8000, log_level="debug", reload=True)
