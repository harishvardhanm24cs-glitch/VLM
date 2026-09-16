from ultralytics import YOLO
import json

def inspect_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        info = {
            'names': model.names
        }
        with open('yolo_audit.json', 'w') as f:
            json.dump(info, f, indent=4)
    except Exception as e:
        with open('yolo_audit.json', 'w') as f:
            json.dump({'error': str(e)}, f, indent=4)

if __name__ == "__main__":
    inspect_yolo_model(r'd:\VLM\models\vehicle_classifier\v_FULL_20260913_232909\best.pt')
