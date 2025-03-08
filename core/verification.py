from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import numpy as np
import os

class DocumentVerifier:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
        
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (250, 160))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray

    def verify_document(self, reference_path, uploaded_path):
        # Load and preprocess both images
        reference_img, reference_gray = self.preprocess_image(reference_path)
        uploaded_img, uploaded_gray = self.preprocess_image(uploaded_path)

        # Compute SSIM
        (score, diff) = structural_similarity(reference_gray, uploaded_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Generate diff visualization
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw contours on uploaded image
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save result image
        result_path = os.path.join(self.upload_folder, "verification_result.png")
        cv2.imwrite(result_path, uploaded_img)

        return {
            "ssim_score": round(score * 100, 2),
            "is_authentic": score >= 0.80,
            "result_image": result_path
        }
