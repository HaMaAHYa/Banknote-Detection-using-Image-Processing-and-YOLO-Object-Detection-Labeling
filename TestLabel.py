import cv2
import os

# --- CONFIGURATION ---
# Path to your images and labels
IMAGE_FOLDER = r"C:\Users\Thanakorn\University\TEST\images"
LABEL_FOLDER = r"C:\Users\Thanakorn\University\TEST\labels"
OUTPUT_FOLDER = r"C:\Users\Thanakorn\University\TEST\testLabels"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Your specific class mapping
CLASS_MAPPING = {
    0: "100 Baht",
    1: "1000 Baht",
    2: "20 Baht",
    3: "50 Baht",
    4: "500 Baht"
}

# Colors for different classes (B, G, R)
COLORS = {
    0: (0, 0, 255),    # Red
    1: (0, 120, 120),  # Yellow
    2: (0, 255, 0),    # Green
    3: (255, 120, 120), # Blue
    4: (255, 0, 255)   # Magenta
}

def denormalize_coordinates(coords, img_width, img_height):
    """Converts YOLO normalized coords to pixel coords."""
    x_center, y_center, w, h = coords
    
    # Calculate absolute pixel values
    w_pixel = int(w * img_width)
    h_pixel = int(h * img_height)
    
    # Calculate top-left corner (x, y)
    x_pixel = int((x_center * img_width) - (w_pixel / 2))
    y_pixel = int((y_center * img_height) - (h_pixel / 2))
    
    return x_pixel, y_pixel, w_pixel, h_pixel

def check_dataset():
    # Get all image files
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images. Processing...")

    for img_file in image_files:
        # Load Image
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error loading image: {img_file}")
            continue
            
        height, width, _ = image.shape
        
        # Load corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(LABEL_FOLDER, label_file)
        
        # --- NEW: Draw Label File Name at Top Left ---
        # Drawn in Red color with thickness 2 for visibility
        #cv2.putText(image, f"Label File: {label_file}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                
                if len(parts) < 5:
                    print(f"Skipping invalid line in {label_file}: {line.strip()}")
                    continue
                
                class_id = int(parts[0])
                coords = list(map(float, parts[1:5]))
                
                # Convert YOLO coords to Pixels
                x, y, w, h = denormalize_coordinates(coords, width, height)
                
                # Get Class Name and Color
                class_name = CLASS_MAPPING.get(class_id, "Unknown")
                color = COLORS.get(class_id, (255, 255, 255))
                
                # Draw Rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # --- Label Background & Text (Top of Box) ---
                (text_w, text_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x, y - 20), (x + text_w, y), color, -1)
                cv2.putText(image, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # --- NEW: Draw Raw Label Data (Bottom of Box) ---
                # We format the numbers to 4 decimal places to keep it readable on screen
                raw_data_str = f"{parts[0]} {float(parts[1]):.4f} {float(parts[2]):.4f} {float(parts[3]):.4f} {float(parts[4]):.4f}"
                
                # Draw black background for raw data legibility
                (raw_w, raw_h), _ = cv2.getTextSize(raw_data_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                # Ensure text doesn't go off bottom edge
                text_y_pos = y + h + 15
                if text_y_pos > height - 5: text_y_pos = y + h - 5 # move inside if too low
                
                cv2.rectangle(image, (x, text_y_pos - 12), (x + raw_w, text_y_pos + 4), (0,0,0), -1) # Black box
                cv2.putText(image, raw_data_str, (x, text_y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) # White text

        else:
            print(f"No label found for {img_file}")
            cv2.putText(image, "NO LABEL FILE FOUND", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save the image with drawn boxes
        output_path = os.path.join(OUTPUT_FOLDER, img_file)
        cv2.imwrite(output_path, image)

    print(f"Done! Check the '{OUTPUT_FOLDER}' folder to inspect your images.")

if __name__ == "__main__":
    check_dataset()