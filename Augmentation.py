import albumentations as A
import cv2
import os
import random
from tqdm import tqdm

# -------- CONFIG --------
IMG_DIR = r"C:\Users\Thanakorn\University\TEST\images"
LBL_DIR = r"C:\Users\Thanakorn\University\TEST\labels"

OUT_IMG_DIR = r"C:\Users\Thanakorn\University\TEST\augment\images"
OUT_LBL_DIR = r"C:\Users\Thanakorn\University\TEST\augment\labels"

NUM_AUG = 100  # TOTAL augmented images
# ------------------------

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

def is_valid_yolo_bbox(bbox, min_size=1e-4):
    x, y, w, h = bbox

    if w <= min_size or h <= min_size:
        return False

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return False

    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False

    return True


# -------- AUGMENT PIPELINE --------
transform = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
    ], p=0.3),

    A.Rotate(
        limit=15,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.15,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    # --- Fold / cut image in half (no occlusion) ---
    A.OneOf([
        A.RandomCropFromBorders(crop_left=0.3, p=1.0),
        A.RandomCropFromBorders(crop_right=0.3, p=1.0),
        A.RandomCropFromBorders(crop_top=0.3, p=1.0),
        A.RandomCropFromBorders(crop_bottom=0.3, p=1.0),
    ], p=0.3),
    
    

    # --- Color & blur only ---
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),

],
bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.7 ,  # 🔥 ensures NO partial boxes
    clip=True,
    check_each_transform=False
))

# -------- LABEL IO --------
def read_yolo_label(path):
    boxes, labels = [], []
    with open(path, "r") as f:
        
        lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            
            if len(parts) < 5:
                print("Skipping invalid line")
                continue
            
            c = int(parts[0])
            x, y, w, h = [round(float(v), 6) for v in parts[1:5]]
            labels.append(int(c))
            boxes.append([x, y, w, h])
            
        
        """
        for line in f:
            c, x, y, w, h = map(float, line.split())
            labels.append(int(c))
            boxes.append([x, y, w, h])
        """
    return boxes, labels

def save_yolo_label(path, boxes, labels):
    with open(path, "w") as f:
        for c, (x, y, w, h) in zip(labels, boxes):
            f.write(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# -------- LOAD IMAGES --------
image_files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print(f"📂 Found {len(image_files)} images")

aug_count = 0
pbar = tqdm(total=NUM_AUG, desc="Augmenting")


while aug_count < NUM_AUG:
    
    try:

        
        img_name = random.choice(image_files)
    
        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))
    
        if not os.path.exists(lbl_path):
            continue
    
        image = cv2.imread(img_path)
        if image is None:
            continue
    
        bboxes, labels = read_yolo_label(lbl_path)
        if len(bboxes) == 0:
            continue
    
        augmented = transform(
        image=image,
        bboxes=bboxes,
        class_labels=labels
        )
        
        # ✅ MANUAL FILTER
        filtered_boxes = []
        filtered_labels = []
        
        for box, lbl in zip(augmented["bboxes"], augmented["class_labels"]):
            if is_valid_yolo_bbox(box):
                filtered_boxes.append(box)
                filtered_labels.append(lbl)
        
        if len(filtered_boxes) == 0:
            continue
    
    
        if len(augmented["bboxes"]) == 0:
            continue
    
        out_img_name = f"{img_name.rsplit('.',1)[0]}_aug_{aug_count}.jpg"
        out_lbl_name = out_img_name.replace(".jpg", ".txt")
    
        cv2.imwrite(
            os.path.join(OUT_IMG_DIR, out_img_name),
            augmented["image"]
        )
    
        save_yolo_label(
            os.path.join(OUT_LBL_DIR, out_lbl_name),
            filtered_boxes,
            filtered_labels
        )
    
        aug_count += 1
        pbar.update(1)
        
    except Exception:
        continue

pbar.close()
print("✅ Total augmentation completed successfully!")
