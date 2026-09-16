import torch
import json

def inspect_model_keys(model_path):
    try:
        model = torch.load(model_path, map_location='cpu')
        
        info = {
            'type': str(type(model)),
            'keys': []
        }
        
        if isinstance(model, dict):
            info['keys'] = list(model.keys())
            for k, v in model.items():
                info[f"key_{k}_type"] = str(type(v))
                if isinstance(v, dict):
                    info[f"key_{k}_keys"] = list(v.keys())
                    if 'names' in v:
                        info[f"key_{k}_names"] = v['names']
                elif hasattr(v, 'names'):
                    info[f"key_{k}_names"] = v.names
                    
        with open('model_keys_audit.json', 'w') as f:
            json.dump(info, f, indent=4)
            
    except Exception as e:
        with open('model_keys_audit.json', 'w') as f:
            json.dump({'error': str(e)}, f, indent=4)

if __name__ == "__main__":
    inspect_model_keys(r'd:\VLM\models\vehicle_classifier\v_FULL_20260913_232909\best.pt')
