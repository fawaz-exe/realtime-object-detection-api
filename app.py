from ultralytics import YOLO
# from ultralytics.models import Detecto 
import cv2
import json

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

for labels in detected_objects_json:

    detected_classes = [obj['class'] for obj in detected_objects]

    string_format = "I can see a {}, also a {}, beside a {} and far away {}"


    if len(detected_classes) == 4:

        formatted_string = string_format.format(*detected_classes)
        print(formatted_string)
    else:
        print("The number of detected objects does not match the string format.")


# print(detected_objects_json)

# detected_objects_string = ()
# detected_objects_string.push(detected_objects_json)
# print(detected_objects_string)



# print(results)
