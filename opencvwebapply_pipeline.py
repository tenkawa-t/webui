# app.py
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)

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
    else:
        raise ValueError(f"Unknown operation: {operation}")

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
        params = step['params']
        image = apply_operation(image, operation, params)

    result_image = encode_image(image)
    return jsonify({'image': result_image})

@app.route('/save_pipeline', methods=['POST'])
def save_pipeline():
    data = request.json
    pipeline_name = data['name']
    pipeline = data['pipeline']
    
    # In a real application, you would save this to a database
    # For this example, we'll just return it as if it was saved
    return jsonify({'name': pipeline_name, 'pipeline': pipeline})

@app.route('/load_pipeline', methods=['GET'])
def load_pipeline():
    # In a real application, you would load this from a database
    # For this example, we'll just return a sample pipeline
    sample_pipeline = [
        {'operation': 'grayscale', 'params': {}},
        {'operation': 'blur', 'params': {'ksize': 5, 'sigmaX': 0}},
        {'operation': 'edge_detection', 'params': {'threshold1': 100, 'threshold2': 200}}
    ]
    return jsonify({'name': 'Sample Pipeline', 'pipeline': sample_pipeline})

if __name__ == '__main__':
    app.run(debug=True)
