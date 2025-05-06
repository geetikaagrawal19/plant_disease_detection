import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

dataset_path = r"D:\Geetika\Niit University\Assignments\3rd year\sem2\Computer vision\plant images"

def segment_disease(image):
   
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([10, 40, 20])  
    upper_bound = np.array([35, 255, 255]) 

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 

    edges = cv2.Canny(mask, 100, 200)
    mask = cv2.bitwise_or(mask, edges)  

    return mask

def classify_plant(image):
    mask = segment_disease(image)

    total_pixels = mask.size
    diseased_pixels = np.sum(mask > 0)
    disease_ratio = (diseased_pixels / total_pixels) * 100

    threshold = 5  

    if disease_ratio > threshold:
        return "Diseased", mask
    else:
        return "Healthy", mask

image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.JPG'))][:5]  

plt.figure(figsize=(10, 10))

for i, file in enumerate(image_files):
    img_path = os.path.join(dataset_path, file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to load image {file}. Skipping...")
        continue

    image = cv2.resize(image, (256, 256)) 

    label, mask = classify_plant(image)

    overlay = cv2.bitwise_and(image, image, mask=mask)

    plt.subplot(len(image_files), 2, 2 * i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original - {label}")
    plt.axis("off")

    plt.subplot(len(image_files), 2, 2 * i + 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Disease Area")
    plt.axis("off")

plt.tight_layout()
plt.show()