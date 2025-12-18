import cv2
import numpy as np
from skimage.feature import hog
from joblib import load
import os

# ----------------------------
# 1. CONFIG
# ----------------------------
MODEL_PATH = r"C:\Users\asus\problem 3\best_damage_model.pkl"
TEST_IMAGE = r"C:\Users\asus\OneDrive\Pictures\Screenshots\Screenshot 2025-12-18 223814.png"

# ----------------------------
# 2. LOAD TRAINED MODEL
# ----------------------------
best_model = load(MODEL_PATH)   # full sklearn Pipeline
print("Model loaded from:", MODEL_PATH)

# ----------------------------
# 3. HOG FEATURE FUNCTION
#    (must match training)
# ----------------------------
def extract_hog(image_gray):
    image_gray = cv2.resize(image_gray, (128, 128))
    features = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

# ----------------------------
# 4. BUILD DEFECT MASK WITH OPENCV
# ----------------------------
def build_defect_mask(gray):
    # equalize contrast
    gray_eq = cv2.equalizeHist(gray)

    # threshold bright + dark anomalies
    thr_bright = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, -10
    )

    thr_dark = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )

    mask = cv2.bitwise_or(thr_bright, thr_dark)

    # clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

# ----------------------------
# 5. MAIN INSPECTION FUNCTION
# ----------------------------
def inspect_image(image_path):
    # load original (color + gray)
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 5.1 defect mask
    mask = build_defect_mask(gray)

    # 5.2 contour boxes for visualization
    out = img_color.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # 5.3 classification with trained model
    feats = extract_hog(gray).reshape(1, -1)
    pred = best_model.predict(feats)[0]
    prob = max(best_model.predict_proba(feats)[0])

    label = "DAMAGED" if pred == 1 else "GOOD"
    color = (0, 0, 255) if pred == 1 else (0, 255, 0)

    cv2.putText(out, f"{label} ({prob:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 5.4 show results
    cv2.imshow("Original", img_color)
    cv2.imshow("Defect Mask", mask)
    cv2.imshow("Result with Boxes + Label", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------
# 6. RUN
# ----------------------------
inspect_image(TEST_IMAGE)
