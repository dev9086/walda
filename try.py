import cv2
import os

# -------------------------------
# IMAGE PATH (CHANGE IF NEEDED)
# -------------------------------
IMAGE_PATH = r"C:\Users\asus\OneDrive\Pictures\Screenshots\Screenshot 2025-12-18 224033.png"push

# -------------------------------
# CHECK IF IMAGE EXISTS
# -------------------------------
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# -------------------------------
# READ IMAGE (COLOR)
# -------------------------------
img_color = cv2.imread(IMAGE_PATH)

if img_color is None:
    raise ValueError("Failed to load image with OpenCV")

# -------------------------------
# CONVERT TO GRAYSCALE
# -------------------------------
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# -------------------------------
# RESIZE (OPTIONAL BUT RECOMMENDED)
# -------------------------------
img_gray = cv2.resize(img_gray, (256, 256))

# -------------------------------
# EDGE DETECTION (CANNY)
# -------------------------------
edges = cv2.Canny(img_gray, 50, 150)

# -------------------------------
# DISPLAY IMAGES
# -------------------------------
cv2.imshow("Original Image (Color)", img_color)
cv2.imshow("Grayscale Image", img_gray)
cv2.imshow("Edge Detection", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
