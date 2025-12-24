import cv2
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMAGE_DIR = r"C:\Users\Thanakorn\University\TEST\images"
LABEL_DIR = r"C:\Users\Thanakorn\University\TEST\labels"
CLASS_ID = 1  # banknote class
MIN_AREA_RATIO = 0.05
SHOW_DEBUG = True   # <-- toggle visualization
# ----------------------------------------

os.makedirs(LABEL_DIR, exist_ok=True)

# Albumentations (NO augmentation)
transform = A.Compose(
    [],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
)

def show_canny(original, edges, title="Canny Edge Detection"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edges")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)   # show for 0.5 sec
    plt.close()

def find_banknote_bbox(image):
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    if SHOW_DEBUG:
        show_canny(image, edges)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (w * h)

    if area_ratio < MIN_AREA_RATIO:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    # YOLO format
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    bw /= w
    bh /= h

    return [cx, cy, bw, bh]

# ----------- PROCESS IMAGES -------------
for img_name in tqdm(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    bbox = find_banknote_bbox(image)
    if bbox is None:
        print(f"❌ No banknote found in {img_name}")
        continue

    transformed = transform(
        image=image,
        bboxes=[bbox],
        class_labels=[CLASS_ID]
    )

    label_path = os.path.join(
        LABEL_DIR, img_name.rsplit(".", 1)[0] + ".txt"
    )

    with open(label_path, "w") as f:
        cx, cy, bw, bh = transformed["bboxes"][0]
        f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

print("✅ Label generation completed!")
