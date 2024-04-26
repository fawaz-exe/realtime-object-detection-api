from flask import Flask, jsonify
from ultralytics import YOLO
import cv2
import json

app = Flask(__name__)

@app.route('/detected_objects', methods=['GET'])
def get_detected_objects():
    model = YOLO('yolov8x.pt')
    results = model.predict(source="0", show=True)

    detected_objects = []

    for result in results.xyxy[0]:
        detected_objects.append({
            'class': result['class'],
            'confidence': result['confidence'],
            'bbox': [result['x1'], result['y1'], result['x2'], result['y2']]
        })

    
    detected_objects_json = json.dumps(detected_objects)

    
    return jsonify(json.loads(detected_objects_json))

if __name__ == '__main__':
    app.run(debug=True)
