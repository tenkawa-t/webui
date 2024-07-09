from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import optuna
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
from ultralytics.models.fastsam import FastSAMPrompt
import logging
import asyncio
import numpy as np
from sklearn.metrics import jaccard_score
import hashlib
from functools import lru_cache

# グローバル変数としてFastSAMモデルを保持
global_fastsam_model = None

fastsam_result = None

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
# 最適化の状態を追跡するためのグローバル変数
optimization_results = {}
optimization_in_progress = set()

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
    # elif operation == 'circle_detection':
    #     return detect_circles(image, params)
    elif operation == 'circle_detection':
                # 画像がグレースケールかカラーかを確認
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                                   dp=params['dp'],
                                   minDist=params['minDist'],
                                   param1=params['param1'],
                                   param2=params['param2'],
                                   minRadius=params['min_radius'],
                                   maxRadius=params['max_radius'])
        
        result_image = image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(result_image, center, 1, (0, 100, 100), 3)
                radius = i[2]
                cv2.circle(result_image, center, radius, (255, 0, 255), 3)
        
        return result_image
    elif operation == 'line_detection':
        return detect_lines(image, params)
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

def detect_circles(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                               dp=params.get('dp', 1),
                               minDist=params.get('minDist', 50),
                               param1=params.get('param1', 200),
                               param2=params.get('param2', 100),
                               minRadius=params.get('min_radius', 0),
                               maxRadius=params.get('max_radius', 0))
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # 円の中心
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # 円周
            radius = i[2]
            cv2.circle(image, center, radius, (255, 0, 255), 3)
    
    return image

def detect_lines(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, params['canny_threshold1'], params['canny_threshold2'])
    lines = cv2.HoughLinesP(edges, params['rho'], params['theta'], params['threshold'],
                            minLineLength=params['minLineLength'], maxLineGap=params['maxLineGap'])
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image

def get_fastsam_result(image):
    global fastsam_result
    if fastsam_result is None:
        try:
            print(f"Input image shape: {image.shape}")
            model = load_fastsam_model()
            results = model(image, device='cpu')
            mask = results[0].masks.data.cpu().numpy().squeeze()
            print(f"FastSAM mask shape: {mask.shape}")
            
            if image.shape[0] > 0 and image.shape[1] > 0:
                # マスクが3次元の場合、最初の次元を取り除く
                if mask.ndim == 3:
                    mask = mask[0]
                
                # マスクのサイズが画像と異なる場合、リサイズする
                if mask.shape != (image.shape[0], image.shape[1]):
                    fastsam_result = cv2.resize((mask > 0).astype(np.uint8), 
                                                (image.shape[1], image.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                else:
                    fastsam_result = (mask > 0).astype(np.uint8)
                
                print(f"Resized FastSAM mask shape: {fastsam_result.shape}")
            else:
                print(f"Invalid image shape: {image.shape}")
                fastsam_result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        except Exception as e:
            print(f"Error in get_fastsam_result_for_optimization: {str(e)}")
            fastsam_result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    return fastsam_result


def circle_detection_to_mask(image, circles):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2], 1, -1)
    return mask



def optimize_parameters(operation, image, n_trials, progress_callback=None):
    def objective(trial):
        params = {}
        if operation == 'blur':
            params['ksize'] = trial.suggest_int('ksize', 3, 15, step=2)
            params['sigmaX'] = trial.suggest_float('sigmaX', 0, 10)
        elif operation == 'edge_detection':
            params['threshold1'] = trial.suggest_int('threshold1', 0, 255)
            params['threshold2'] = trial.suggest_int('threshold2', 0, 255)
        elif operation == 'threshold':
            params['thresh'] = trial.suggest_int('thresh', 0, 255)
            params['maxval'] = trial.suggest_int('maxval', 0, 255)
            params['type'] = trial.suggest_categorical('type', [0, 1, 2, 3, 4])
        elif operation == 'circle_detection':
            global fastsam_result
            fastsam_result = None  # FastSAM の結果をリセット
            
            fastsam_mask = get_fastsam_result(image)
            params['dp'] = trial.suggest_float('dp', 1, 3)
            params['minDist'] = trial.suggest_int('minDist', 10, 100)
            params['param1'] = trial.suggest_int('param1', 50, 300)
            params['param2'] = trial.suggest_int('param2', 50, 300)
            params['min_radius'] = trial.suggest_int('min_radius', 0, 50)
            params['max_radius'] = trial.suggest_int('max_radius', 50, 200)
            return evaluate_circle_detection(image, params, fastsam_mask)
        elif operation == 'line_detection':
            params['canny_threshold1'] = trial.suggest_int('canny_threshold1', 0, 255)
            params['canny_threshold2'] = trial.suggest_int('canny_threshold2', 0, 255)
            params['rho'] = trial.suggest_int('rho', 1, 10)
            params['theta'] = trial.suggest_float('theta', 0, np.pi/2)
            params['threshold'] = trial.suggest_int('threshold', 10, 200)
            params['minLineLength'] = trial.suggest_int('minLineLength', 10, 200)
            params['maxLineGap'] = trial.suggest_int('maxLineGap', 1, 50)
        
        processed_image = apply_operation(image.copy(), operation, params)
        return evaluate_result(image, processed_image, operation)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback] if progress_callback else None)
    return study.best_params

def evaluate_circle_detection(image, params, fastsam_mask):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                                   dp=params['dp'],
                                   minDist=params['minDist'],
                                   param1=params['param1'],
                                   param2=params['param2'],
                                   minRadius=params['min_radius'],
                                   maxRadius=params['max_radius'])
        
        circle_mask = circle_detection_to_mask(image, circles)
        
        if fastsam_mask.shape != circle_mask.shape:
            print(f"Shape mismatch: fastsam_mask {fastsam_mask.shape}, circle_mask {circle_mask.shape}")
            fastsam_mask = cv2.resize(fastsam_mask, (circle_mask.shape[1], circle_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        iou = jaccard_score(fastsam_mask.flatten(), circle_mask.flatten())
        return iou
    except Exception as e:
        print(f"Error in evaluate_circle_detection: {str(e)}")
        return 0  # エラーが発生した場合は最低スコアを返す

def evaluate_result(original_image, processed_image, operation):
    if operation == 'blur':
        return -cv2.Laplacian(processed_image, cv2.CV_64F).var()  # 鮮明度の逆数
    elif operation == 'edge_detection':
        return np.mean(processed_image)  # エッジの平均強度
    elif operation == 'threshold':
        return -np.std(processed_image)  # 標準偏差の逆数（コントラストの指標）
    elif operation == 'circle_detection' or operation == 'line_detection':
        # 検出された特徴の数を評価
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, min_radius=0, max_radius=0)
        return len(circles[0]) if circles is not None else 0
    else:
        return 0  # デフォルトの評価スコア

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
    elif operation == 'circle_detection':
        code = f"""gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp={params['dp']}, minDist={params['minDist']},
                                        param1={params['param1']}, param2={params['param2']},
                                        minRadius={params['min_radius']}, maxRadius={params['max_radius']})
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        cv2.circle(image, center, 1, (0, 100, 100), 3)
                        radius = i[2]
                        cv2.circle(image, center, radius, (255, 0, 255), 3)"""
    elif operation == 'line_detection':
        code = f"""gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, {params['canny_threshold1']}, {params['canny_threshold2']})
                lines = cv2.HoughLinesP(edges, {params['rho']}, {params['theta']}, {params['threshold']},
                                        minLineLength={params['minLineLength']}, maxLineGap={params['maxLineGap']})
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)"""
    else:
        code = f"# Unknown operation: {operation}"
    
    return f"{comment}\n{code}"

def get_nearest_object(image, x, y, max_distance=50):
    try:
        logger.info(f"Processing request for coordinates ({x}, {y})")
        
        # Preprocess the image
        image = preprocess_image(image)
        
        # Calculate image hash for caching
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        # Check cache
        cached_result = get_cached_result(image_hash)
        if cached_result is not None:
            logger.info("Using cached FastSAM result")
            masks_np = cached_result
        else:
            logger.info("Processing new image with FastSAM")
            model = get_fastsam_model()
            results = model(image, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            
            logger.info("Creating FastSAM prompt")
            prompt_process = FastSAMPrompt(image, results, device='cpu')
            
            logger.info("Getting all objects")
            ann = prompt_process.everything_prompt()
            
            logger.info(f"Type of ann: {type(ann)}")
            
            if isinstance(ann, list) and len(ann) > 0 and hasattr(ann[0], 'masks'):
                masks = ann[0].masks
                logger.info(f"Type of masks: {type(masks)}")
                
                # Convert Masks object to numpy array
                if hasattr(masks, 'data'):
                    masks_np = masks.data.cpu().numpy()
                elif hasattr(masks, 'numpy'):
                    masks_np = masks.numpy()
                else:
                    logger.error(f"Unexpected type for masks: {type(masks)}")
                    return None
                
                # Cache the result
                set_cached_result(image_hash, masks_np)
            else:
                logger.error(f"Unexpected structure for ann: {type(ann)}")
                return None
        
        logger.info(f"Shape of masks_np: {masks_np.shape}")
        
        nearest_object = None
        min_distance = float('inf')
        
        for i in range(masks_np.shape[0]):
            mask = masks_np[i]
            logger.info(f"Processing mask {i}, shape: {mask.shape}")
            
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            if mask.ndim != 2:
                logger.warning(f"Skipping mask with unexpected shape: {mask.shape}")
                continue
            
            where_result = np.where(mask)
            if len(where_result) != 2:
                logger.warning(f"Unexpected where result shape: {len(where_result)}")
                continue
            
            coords = np.column_stack(where_result)
            distances = np.sqrt(np.sum((coords - np.array([y, x]))**2, axis=1))
            min_dist = np.min(distances) if distances.size > 0 else float('inf')
            
            if min_dist < min_distance and min_dist <= max_distance:
                min_distance = min_dist
                nearest_object = mask
        
        if nearest_object is not None:
            logger.info("Nearest object found")
            # オブジェクトの中心を計算
            center = np.mean(np.argwhere(nearest_object), axis=0)
            center = (int(center[1]), int(center[0]))  # (x, y) 形式に変換
            # SelectedRegion オブジェクトとして保存
            selected_regions.append(SelectedRegion(nearest_object, center))
            
            logger.info("Applying mask to image")
            masked_image = cv2.bitwise_and(image, image, mask=nearest_object.astype(np.uint8))
            return masked_image
        else:
            logger.info("No object found near the specified coordinates")
            return None
    except Exception as e:
        logger.error(f"Error in get_nearest_object: {str(e)}", exc_info=True)
        raise

def draw_selected_regions(image):
    result_image = image.copy()
    for i, region in enumerate(selected_regions):
        mask = region.mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
        cv2.putText(result_image, f"Region {i+1}", region.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return result_image

class SelectedRegion:
    def __init__(self, mask, center):
        self.mask = mask
        self.center = center

selected_regions = []

def add_selected_region(mask, center):
    selected_regions.append(SelectedRegion(mask, center))

def get_selected_regions():
    return selected_regions

@app.post("/clear_selected_regions")
async def clear_selected_regions_endpoint():
    global selected_regions
    selected_regions = []
    return JSONResponse(content={'message': 'All selected regions have been cleared.'})


@app.post("/get_nearest_object")
async def get_nearest_object_endpoint(data: dict):
    try:
        image_data = data['image']
        x = data['x']
        y = data['y']
        
        image = decode_image(image_data)
        logger.info(f"Image shape: {image.shape}, Coordinates: ({x}, {y})")
        
        result = get_nearest_object(image, x, y)
        
        if result is not None:
            # 選択された領域を含む画像を作成
            result_with_regions = draw_selected_regions(image)
            result_image = encode_image(result_with_regions)
            return JSONResponse(content={'image': result_image})
        else:
            return JSONResponse(content={'message': 'No object found near the specified coordinates.'})
    except Exception as e:
        logger.error(f"Error in get_nearest_object_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def apply_optimized_circle_detection(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                               dp=params['dp'],
                               minDist=params['minDist'],
                               param1=params['param1'],
                               param2=params['param2'],
                               minRadius=params['minRadius'],
                               maxRadius=params['maxRadius'])
    
    result_image = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(result_image, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv2.circle(result_image, center, radius, (255, 0, 255), 3)
    
    return result_image

@app.post("/optimize_and_detect_circles")
async def optimize_and_detect_circles(data: dict):
    try:
        image_data = data['image']
        image = decode_image(image_data)
        
        if not selected_regions:
            return JSONResponse(content={'error': 'No regions selected. Please select regions first.'})
        
        best_params = optimize_circle_detection(image, selected_regions)
        result_image = apply_optimized_circle_detection(image, best_params)
        
        result_image_encoded = encode_image(result_image)
        return JSONResponse(content={'image': result_image_encoded, 'params': best_params})
    except Exception as e:
        logger.error(f"Error in optimize_and_detect_circles: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def optimize_circle_detection(image, selected_regions, n_trials=100):
    def objective(trial):
        dp = trial.suggest_float('dp', 1.0, 3.0)
        minDist = trial.suggest_int('minDist', 20, 100)
        param1 = trial.suggest_int('param1', 50, 300)
        param2 = trial.suggest_int('param2', 10, 100)
        minRadius = trial.suggest_int('minRadius', 0, 50)
        maxRadius = trial.suggest_int('maxRadius', 50, 200)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                   param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)
        
        if circles is None:
            return 0

        score = 0
        for region in selected_regions:
            region_mask = region.mask
            for circle in circles[0]:
                x, y, r = circle
                circle_mask = np.zeros_like(region_mask)
                cv2.circle(circle_mask, (int(x), int(y)), int(r), 1, -1)
                
                intersection = np.logical_and(region_mask, circle_mask)
                union = np.logical_or(region_mask, circle_mask)
                iou = np.sum(intersection) / np.sum(union)
                score += iou

        return score / len(selected_regions)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params

@app.post("/show_selected_regions")
async def show_selected_regions(data: dict):
    try:
        image_data = data['image']
        image = decode_image(image_data)
        
        if not selected_regions:
            return JSONResponse(content={'error': 'No regions selected. Please select regions first.'})
        
        result_image = draw_selected_regions(image, selected_regions)
        result_image_encoded = encode_image(result_image)
        return JSONResponse(content={'image': result_image_encoded})
    except Exception as e:
        logger.error(f"Error in show_selected_regions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.post("/optimize_parameters")
async def optimize_parameters_endpoint(request: Request, background_tasks: BackgroundTasks, data: dict = Body(...)):
    logger.info("Received request to /optimize_parameters")
    try:
        logger.info(f"Request data: {data.keys()}")
        print(f"Request data: {data.keys()}") 
        image_data = data['image']
        operation = data['operation']
        n_trials = data.get('n_trials', 10)  # デフォルト値を10に設定

        if operation in optimization_in_progress:
            logger.warning(f"Optimization already in progress for operation: {operation}")
            print(f"Optimization already in progress for operation: {operation}")
            return JSONResponse(content={'message': 'Optimization already in progress for this operation'})

        image = decode_image(image_data)
        print(f"Starting optimization for operation: {operation}")
        logger.info(f"Starting optimization for operation: {operation}")
        background_tasks.add_task(run_optimization, operation, image, n_trials)
        optimization_in_progress.add(operation)

        return JSONResponse(content={'message': 'Optimization started'})
    except Exception as e:
        logger.error(f"Error in optimize_parameters_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# グローバル変数に進捗情報を追加
optimization_progress = {}
async def run_optimization(operation, image, n_trials):
    global optimization_results, optimization_progress
    try:
        logger.info(f"Starting optimization for operation: {operation}")
        optimization_progress[operation] = 0
        
        def objective(trial):
            params = {}
            if operation == 'blur':
                params['ksize'] = trial.suggest_int('ksize', 3, 15, step=2)
                params['sigmaX'] = trial.suggest_float('sigmaX', 0, 10)
            elif operation == 'edge_detection':
                params['threshold1'] = trial.suggest_int('threshold1', 0, 255)
                params['threshold2'] = trial.suggest_int('threshold2', 0, 255)
            elif operation == 'threshold':
                params['thresh'] = trial.suggest_int('thresh', 0, 255)
                params['maxval'] = trial.suggest_int('maxval', 0, 255)
                params['type'] = trial.suggest_categorical('type', [0, 1, 2, 3, 4])
            elif operation == 'circle_detection':
                global fastsam_result
                fastsam_result = None  # FastSAM の結果をリセット
                
                fastsam_mask = get_fastsam_result(image)
                params['dp'] = trial.suggest_float('dp', 1, 3)
                params['minDist'] = trial.suggest_int('minDist', 10, 100)
                params['param1'] = trial.suggest_int('param1', 50, 300)
                params['param2'] = trial.suggest_int('param2', 50, 300)
                params['min_radius'] = trial.suggest_int('min_radius', 0, 50)
                params['max_radius'] = trial.suggest_int('max_radius', 50, 200)
                return evaluate_circle_detection(image, params, fastsam_mask)
            elif operation == 'line_detection':
                params['canny_threshold1'] = trial.suggest_int('canny_threshold1', 0, 255)
                params['canny_threshold2'] = trial.suggest_int('canny_threshold2', 0, 255)
                params['rho'] = trial.suggest_int('rho', 1, 10)
                params['theta'] = trial.suggest_float('theta', 0, np.pi/2)
                params['threshold'] = trial.suggest_int('threshold', 10, 200)
                params['minLineLength'] = trial.suggest_int('minLineLength', 10, 200)
                params['maxLineGap'] = trial.suggest_int('maxLineGap', 1, 50)
            
            processed_image = apply_operation(image.copy(), operation, params)
            return evaluate_result(image, processed_image, operation)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        optimization_results[operation] = best_params
        optimization_progress[operation] = 100
        logger.info(f"Optimization completed for operation: {operation}")
    except Exception as e:
        logger.error(f"Error during optimization for operation {operation}: {str(e)}", exc_info=True)
    finally:
        optimization_in_progress.discard(operation)

@app.get("/optimization_result/{operation}")
async def get_optimization_result(operation: str):
    logger.info(f"Retrieving optimization result for operation: {operation}")
    logger.info(f"Current progress: {optimization_progress.get(operation, 0)}")
    if operation in optimization_results:
        optimization_in_progress.discard(operation)
        return JSONResponse(content={'status': 'completed', 'best_params': optimization_results[operation], 'progress': 100})
    elif operation in optimization_in_progress:
        progress = optimization_progress.get(operation, 0)
        logger.info(f"Optimization in progress. Current progress: {progress}")
        return JSONResponse(content={'status': 'in_progress', 'progress': progress})
    else:
        logger.info(f"Optimization not started for operation: {operation}")
        return JSONResponse(content={'status': 'not_started', 'progress': 0})
selected_objects = []
cache = {}

def get_fastsam_model():
    global global_fastsam_model
    if global_fastsam_model is None:
        global_fastsam_model = FastSAM('FastSAM-x.pt')
    return global_fastsam_model

def get_cached_result(image_hash):
    return cache.get(image_hash)

def set_cached_result(image_hash, result):
    cache[image_hash] = result

def preprocess_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)

def process_image_with_fastsam(image):
    model = get_fastsam_model()
    results = model(image, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    prompt_process = FastSAMPrompt(image, results, device='cpu')
    ann = prompt_process.everything_prompt()
    
    if isinstance(ann, list) and len(ann) > 0 and hasattr(ann[0], 'masks'):
        masks = ann[0].masks
        if hasattr(masks, 'data'):
            return masks.data.cpu().numpy()
        elif hasattr(masks, 'numpy'):
            return masks.numpy()
        else:
            logger.error(f"Unexpected type for masks in process_image_with_fastsam: {type(masks)}")
            return None
    else:
        logger.error(f"Unexpected structure for ann in process_image_with_fastsam: {type(ann)}")
        return None
    
def preprocess_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)

    
@app.get("/routes")
async def list_routes():
    routes = [
        {
            "path": route.path,
            "name": route.name,
            "methods": [method for method in route.methods]
        }
        for route in app.routes
    ]
    return {"routes": routes}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/debug")
async def debug_endpoint(request: Request):
    data = await request.json()
    print(f"Received data at /debug: {data}")
    return {"status": "success"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.get("/test")
async def root():
    return {"message": "Hello, World!"}


if __name__ == '__main__':
    import uvicorn
    print("Available routes:")
    for route in app.routes:
        print(f"{route.path} [{', '.join(route.methods)}]")
    uvicorn.run("complete-python-backend2:app", host="127.0.0.1", port=8000, log_level="debug", reload=True)
