import cv2
import numpy as np
import os

# path to the image you want to check
TEST_IMAGE = r"C:\path\to\your\panel_image.png"

def check_panel(image_path):
    # 1) load and show image
    img_color = cv2.imread(image_path)          # color for display
    if img_color is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    cv2.imshow("Input image", img_color)
    cv2.waitKey(1)   # show quickly so window appears

    # 2) extract HOG features (same as training)
    feats = extract_hog(image_path).reshape(1, -1)

    # 3) predict with best model
    pred = best_damage_model.predict(feats)[0]
    prob = max(best_model.predict_proba(feats)[0])

    label = "DAMAGED" if pred == 1 else "GOOD"
    color = (0, 0, 255) if pred == 1 else (0, 255, 0)

    # 4) put text on image and show
    out = img_color.copy()
    cv2.putText(out, f"{label} ({prob:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

check_panel(TEST_IMAGE)
