import os
import pydicom
import cv2
import pandas as pd
from tqdm import tqdm

# Paths
DICOM_DIR = "lesson3-data//stage_2_train_images"
OUTPUT_DIR = "png_images/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load labels
df = pd.read_csv("lesson3-data//stage_2_train_labels.csv")
image_ids = df["patientId"].unique()

for pid in tqdm(image_ids):
    dcm_path = os.path.join(DICOM_DIR, pid + ".dcm")
    dcm_data = pydicom.dcmread(dcm_path)
    img = dcm_data.pixel_array

    # Normalize to [0, 255]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(os.path.join(OUTPUT_DIR, pid + ".png"), img)