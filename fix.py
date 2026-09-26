import re
import os

fixes = [
    (
        r"D:\VLM\run_full_pipeline.py",
        "for line in process.stdout:\n",
        "for line in process.stdout:\n            pass\n",
    ),
    (
        r"D:\VLM\run_phase10_validation.py",
        "if fmr < 1.0 and fnmr < 5.0:\n    else:",
        "if fmr < 1.0 and fnmr < 5.0:\n        pass\n    else:",
    ),
    (
        r"D:\VLM\test_am_logic.py",
        "async def broadcast(self, msg):\nmanager",
        "async def broadcast(self, msg):\n        pass\nmanager",
    ),
    (
        r"D:\VLM\test_cameras.py",
        "if cap is None or not cap.isOpened():\n        else:",
        "if cap is None or not cap.isOpened():\n            pass\n        else:",
    ),
    (
        r"D:\VLM\test_esp32_integration.py",
        'if "status" in sensors and sensors["status"] == "OFFLINE":\n    else:',
        'if "status" in sensors and sensors["status"] == "OFFLINE":\n        pass\n    else:',
    ),
    (
        r"D:\VLM\test_event_append.py",
        "if state.events:\n    else:",
        "if state.events:\n        pass\n    else:",
    ),
    (
        r"D:\VLM\test_heic.py",
        "except ImportError:\n\nimg_path",
        "except ImportError:\n    pass\n\nimg_path",
    ),
    (
        r"D:\VLM\test_phase1.py",
        "if abs(total_prob - 1.0) < 0.1:\n    else:",
        "if abs(total_prob - 1.0) < 0.1:\n        pass\n    else:",
    ),
    (
        r"D:\VLM\tools\test_vehicle_classifier.py",
        "if conf < min_confidence:\n        elif",
        "if conf < min_confidence:\n            pass\n        elif",
    ),
    (
        r"D:\VLM\suspicious_activity.py",
        "if err and args.debug:\n    print(json",
        "if err and args.debug:\n        pass\n    print(json",
    ),
    (
        r"D:\VLM\VLM_Surveillance_Project\main.py",
        'if v_class == "Military" and v_sub:\n                                                    elif',
        'if v_class == "Military" and v_sub:\n                                                        pass\n                                                    elif',
    ),
]

for file, search, replace in fixes:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
    if search in content:
        content = content.replace(search, replace)
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {file}")
    else:
        print(f"Could not fix {file}")
