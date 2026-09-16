import torch
import json
import sys

def inspect_model(model_path):
    try:
        model = torch.load(model_path, map_location='cpu')
        
        info = {}
        info['type'] = str(type(model))
        
        if isinstance(model, dict):
            if 'model' in model:
                m = model['model']
                info['model_type'] = str(type(m))
                if hasattr(m, 'names'):
                    info['names'] = m.names
                elif hasattr(m, 'module') and hasattr(m.module, 'names'):
                    info['names'] = m.module.names
            
            # If standard yolov8 dict contains 'nc' and 'names'
            if 'nc' in model:
                info['nc'] = model['nc']
            if 'names' in model:
                info['names'] = model['names']
        elif hasattr(model, 'names'):
            info['names'] = model.names
            
        with open('model_audit.json', 'w') as f:
            json.dump(info, f, indent=4)
            
    except Exception as e:
        with open('model_audit.json', 'w') as f:
            json.dump({'error': str(e)}, f, indent=4)

if __name__ == "__main__":
    inspect_model(r'd:\VLM\models\vehicle_classifier\v_FULL_20260913_232909\best.pt')
