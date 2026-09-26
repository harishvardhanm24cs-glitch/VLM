import cv2
import sys

try:
    from PIL import Image
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

img_path = r"D:\VLM\PSNA BUS\IMG_2963.HEIC"
img_cv = cv2.imread(img_path)
if img_cv is not None:
    pass
else:
    pass
try:
    img_pil = Image.open(img_path)
except Exception as e:
    pass
