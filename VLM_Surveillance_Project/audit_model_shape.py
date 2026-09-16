import torch
import json

def inspect_shape(model_path):
    try:
        model = torch.load(model_path, map_location='cpu')
        
        info = {}
        if 'classifier.1.weight' in model:
            shape = list(model['classifier.1.weight'].shape)
            info['classifier_shape'] = shape
            info['num_classes'] = shape[0]
        elif 'fc.weight' in model:
            shape = list(model['fc.weight'].shape)
            info['num_classes'] = shape[0]
            
        with open('model_shape_audit.json', 'w') as f:
            json.dump(info, f, indent=4)
            
    except Exception as e:
        with open('model_shape_audit.json', 'w') as f:
            json.dump({'error': str(e)}, f, indent=4)

if __name__ == "__main__":
    inspect_shape(r'd:\VLM\models\vehicle_classifier\v_FULL_20260913_232909\best.pt')
