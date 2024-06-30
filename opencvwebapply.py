# app.py

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import torch
import torchvision
from PIL import Image
import io
import timm
import json

app = Flask(__name__)

def load_yolov5_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def load_yolox_model():
    return torch.hub.load('Megvii-BaseDetection/YOLOX', 'yolox_s', pretrained=True)

def load_detr_model():
    return torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

def load_centernet_model():
    return torch.hub.load('pytorch/vision:v0.10.0', 'centernet_resnet50_fpn', pretrained=True)

def load_fcos_model():
    return torch.hub.load('pytorch/vision:v0.10.0', 'fcos_resnet50_fpn', pretrained=True)

def load_unet_model():
    return torchvision.models.segmentation.fcn_resnet50(pretrained=True)

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
        return yolov5(image)
    elif operation == 'yolox':
        return yolox(image)
    elif operation == 'detr':
        return detr(image)
    elif operation == 'centernet':
        return centernet(image)
    elif operation == 'fcos':
        return fcos(image)
    elif operation == 'segmentation':
        return segmentation(image)
    elif operation == 'draw_boxes':
        return draw_boxes(image, params['boxes'])
    elif operation == 'apply_mask':
        return apply_mask(image, params['mask'])
    else:
        raise ValueError(f"Unknown operation: {operation}")

def yolov5(image):
    model = load_yolov5_model()
    results = model(image)
    return results.render()[0]

def yolox(image):
    model = load_yolox_model()  
    results = model(image)
    return draw_results(image, results)

def detr(image):
    model = load_detr_model()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)
    return draw_results(image, prediction)

def centernet(image):
    model = load_centernet_model()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)[0]
    return draw_results(image, prediction)

def fcos(image):  
    model = load_fcos_model()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)[0]
    return draw_results(image, prediction)

def segmentation(image):
    model = load_unet_model() 
    input_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = torchvision.transforms.functional.to_tensor(input_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return (output_predictions > 0).astype(np.uint8) * 255

def draw_results(image, results):
    if 'boxes' in results:
        for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
            if score > 0.5:
                box = [int(i) for i in box.tolist()]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {score:.2f}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    elif 'xyxy' in results:
        for det in results.xyxy[0]:
            if det[4] > 0.5:
                box = [int(i) for i in det[:4]]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{det[5]}: {det[4]:.2f}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def draw_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    image_data = data['image']
    pipeline = data['pipeline']

    image = decode_image(image_data)
    
    for step in pipeline:
        operation = step['operation']
        params = step.get('params', {})
        try:
            image = apply_operation(image, operation, params)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    result_image = encode_image(image)
    return jsonify({'image': result_image})

@app.route('/save_pipeline', methods=['POST'])
def save_pipeline():
    data = request.json
    pipeline_name = data['name']
    pipeline = data['pipeline']
    
    with open(f"{pipeline_name}.json", "w") as f:
        json.dump(pipeline, f)

    return jsonify({'name': pipeline_name, 'pipeline': pipeline})

@app.route('/load_pipeline', methods=['GET'])
def load_pipeline():
    pipeline_name = request.args.get('name')

    try:
        with open(f"{pipeline_name}.json", "r") as f:
            pipeline = json.load(f)
    except FileNotFoundError:
        return jsonify({'error': 'Pipeline not found'}), 404

    return jsonify({'name': pipeline_name, 'pipeline': pipeline})

if __name__ == '__main__':
    app.run(debug=True)