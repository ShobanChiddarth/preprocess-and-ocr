import numpy as np
import imutils
import cv2

def process_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist = (dist * 255).astype("uint8")

    dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)

    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    chars = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 35 and h >= 100:
            chars.append(c)
    
    chars = np.vstack([chars[i] for i in range(0, len(chars))])
    hull = cv2.convexHull(chars)
    
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.dilate(mask, None, iterations=2)

    final = cv2.bitwise_and(opening, opening, mask=mask)

    success, encoded_bytes = cv2.imencode(".png", final)

    if success:
        return encoded_bytes.tobytes()
    else:
        return b''
