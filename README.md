# 💵 Banknote Detection Dataset Generator (YOLO Format)

This project provides a **complete pipeline for creating a YOLO-ready banknote detection dataset** using traditional image processing + data augmentation.

## 📌 What This Project Does

This repository helps you:

1. **Automatically generate YOLO labels** from raw banknote images
2. **Augment images + bounding boxes** safely using Albumentations
3. **Visually verify labels** to ensure correctness before training

✔ No manual labeling
✔ YOLO-compatible
✔ Beginner friendly

---

## 🧠 Pipeline Overview

```
Raw Images
   ↓
Edge Detection (Canny)
   ↓
Largest Contour Detection
   ↓
YOLO Bounding Box Generation
   ↓
Data Augmentation (with bbox safety)
   ↓
Visual Label Validation
   ↓
YOLO-Ready Dataset 🚀
```

---

## 📂 Project Structure

```
banknote-detection/
│
├── images/                 # Original images
├── labels/                 # Auto-generated YOLO labels
│
├── augment/
│   ├── images/             # Augmented images
│   └── labels/             # Augmented labels
│
├── testLabels/             # Visualized bounding boxes
│
├── Labeling.py             # Auto label generation
├── Augmentation.py         # Dataset augmentation
├── TestLabel.py            # Label visualization & validation
│
└── README.md
```

---

## ⚙️ Requirements

Install required Python packages:

```bash
pip install opencv-python numpy albumentations tqdm matplotlib
```

✔ Python 3.8+ recommended
✔ Works on Windows / Linux / macOS

---

## 🚀 Step-by-Step Usage (Beginner Friendly)

---

### 🟢 Step 1: Prepare Your Images

Put all banknote images inside:

```
images/
```

Supported formats:

* `.jpg`
* `.png`
* `.jpeg`

⚠️ Images should contain **one dominant banknote** per image.

---

### 🟢 Step 2: Auto-Generate YOLO Labels

Run:

```bash
python Labeling.py
```

### 🔍 What This Script Does

* Converts image to grayscale
* Applies Gaussian blur + Canny edge detection
* Finds the **largest contour**
* Converts bounding box to **YOLO format**
* Saves `.txt` label files

📄 Output example (`labels/image1.txt`):

```
1 0.512345 0.498732 0.623456 0.312345
```

### 🧠 Notes

* `MIN_AREA_RATIO` filters noise
* `SHOW_DEBUG = True` shows edge detection visualization

---

### 🟢 Step 3: Augment the Dataset (Highly Recommended)

Run:

```bash
python Augmentation.py
```

### ✨ Augmentations Applied

* Flip (horizontal / vertical)
* Rotation
* Shift & scale
* Border cropping (simulates folding)
* Brightness & contrast
* Blur & color shifts

### 🛡️ Bounding Box Safety

* Invalid boxes are discarded
* Partial objects are **automatically rejected**
* YOLO normalization preserved

📦 Output folders:

```
augment/images/
augment/labels/
```

---

### 🟢 Step 4: Visualize & Verify Labels (VERY IMPORTANT)

Run:

```bash
python TestLabel.py
```

### 👀 What You Will See

* Bounding boxes drawn on images
* Class name displayed
* Raw YOLO values shown
* Color-coded per banknote type

📁 Output:

```
testLabels/
```

⚠️ **Always check this before training YOLO**

---

## 🏷️ Class Mapping Example

```python
CLASS_MAPPING = {
    0: "100 Baht",
    1: "1000 Baht",
    2: "20 Baht",
    3: "50 Baht",
    4: "500 Baht"
}
```

🎨 Each class has a unique bounding box color.

---

## 🔧 Customization Guide

### Change Dataset Paths

Edit paths at the top of each script:

```python
IMAGE_DIR = r"path/to/images"
LABEL_DIR = r"path/to/labels"
```

---

### Change Number of Augmented Images

```python
NUM_AUG = 100
```

---

### Change YOLO Class ID

```python
CLASS_ID = 1
```

---

## 🧪 Tested Use Cases

✔ YOLOv5
✔ YOLOv8
✔ Custom object detection projects
✔ Academic / student projects

---

## ⚠️ Limitations

* Assumes **one main banknote per image**
* Works best when background is not too cluttered
* Not designed for multi-object scenes (yet)

---

## 🚧 Future Improvements

* Multi-banknote detection
* Automatic denomination classification
* Integration with YOLO training scripts
* GUI for beginners

---

## 🤝 Contributing

Contributions are welcome!

* Improve detection logic
* Add more augmentations
* Extend to other documents (IDs, cards)

---

## 📜 License

This project is open-source and free for **educational and research use**.

---

## ⭐ Final Advice for Beginners

> **Never train YOLO without visualizing your labels first.**
> `TestLabel.py` can save you weeks of debugging.
