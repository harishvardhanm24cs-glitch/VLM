import os

fixes = [
    (
        r"D:\VLM\test_esp32_integration.py",
        "    else:\n\n\n    print(",
        "    else:\n        pass\n\n    print(",
    ),
    (
        r"D:\VLM\test_event_append.py",
        "    else:\n\nif __name__",
        "    else:\n        pass\n\nif __name__",
    ),
    (
        r"D:\VLM\test_heic.py",
        "if img_cv is not None:\nelse:",
        "if img_cv is not None:\n    pass\nelse:",
    ),
    (
        r"D:\VLM\test_phase1.py",
        "    else:\n\nfor i in range(5):",
        "    else:\n        pass\n\nfor i in range(5):",
    ),
    (
        r"D:\VLM\test_cameras.py",
        "    if available:\n    else:",
        "    if available:\n        pass\n    else:",
    ),
    (
        r"D:\VLM\tools\test_vehicle_classifier.py",
        '        elif predicted_class == "army_vehicle":\n        elif predicted_class == "normal_vehicle":\n        else:',
        '        elif predicted_class == "army_vehicle":\n            pass\n        elif predicted_class == "normal_vehicle":\n            pass\n        else:',
    ),
    (
        r"D:\VLM\run_phase10_validation.py",
        "    else:\n\nif __name__",
        "    else:\n        pass\n\nif __name__",
    ),
    (
        r"D:\VLM\VLM_Surveillance_Project\main.py",
        '                                                    elif v_class == "Normal" and v_sub:\n\n                                        except',
        '                                                    elif v_class == "Normal" and v_sub:\n                                                        pass\n\n                                        except',
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
        print(f"Failed to find exact text in {file}")
