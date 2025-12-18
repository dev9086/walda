import cv2
import numpy as np
import glob
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#resize and convert it gray
def preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path)           # read
    img = cv2.resize(img, size)          # make size fixed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # small blur removes sensor noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, gray

#conver crack and spot with open cv
def detect_defect_mask(gray):
    # normalize contrast
    gray_norm = cv2.equalizeHist(gray)

    # get bright regions (hot spots, reflections)
    th_bright = cv2.adaptiveThreshold(
        gray_norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, -10
    )

    # get dark regions (burn marks, cracks)
    th_dark = cv2.adaptiveThreshold(
        gray_norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )

    mask = cv2.bitwise_or(th_bright, th_dark)

    # remove tiny dots (noise)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

# drawing dots and marker
def draw_defects(img, mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)   # red box
        cv2.circle(img, (x + w//2, y + h//2), 3, (0, 255, 0), -1)  # green dot
    return img

#running of the code
img_path = r"C:\Users\asus\OneDrive\Pictures\Screenshots\Screenshot 2025-12-18 175242.png"
img, gray = preprocess(img_path)
mask = detect_defect_mask(gray)
marked = draw_defects(img.copy(), mask)

cv2.imshow("mask", mask)
cv2.imshow("marked", marked)
cv2.waitKey(0)
cv2.destroyAllWindows()


