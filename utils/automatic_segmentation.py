
import cv2
import numpy as np
import os
import glob
from collections import defaultdict


HSV_RANGES = {
    'acerola': {
        'W': {'lower': np.array([138, 52, 0]), 'upper': np.array([179, 255, 255])},
        'B': {'lower': np.array([0, 71, 48]), 'upper': np.array([179, 255, 255])}
    },
    'lemon': {
        'W': {'lower': np.array([0, 46, 0]), 'upper': np.array([59, 255, 255])},
        'B': {'lower': np.array([0, 72, 49]), 'upper': np.array([177, 255, 255])}
    },
    'cherry_tomato': {
        'W': {'lower': np.array([0, 78, 67]), 'upper': np.array([79, 255, 255])},
        'B': {'lower': np.array([0, 51, 190]), 'upper': np.array([179, 255, 255])}
    },
    'khaki': {
        'W': {'lower': np.array([0, 76, 0]), 'upper': np.array([85, 255, 255])},
        'B': {'lower': np.array([0, 66, 61]), 'upper': np.array([179, 255, 255])}
    },
    'banana': {
        'W': {'lower': np.array([0, 62, 0]), 'upper': np.array([179, 255, 255])},
        'B': {'lower': np.array([0, 62, 69]), 'upper': np.array([179, 255, 255])}
    },
    'lime': {
        'W': {'lower': np.array([0, 52, 0]), 'upper': np.array([179, 255, 255])},
        'B': {'lower': np.array([0, 26, 73]), 'upper': np.array([179, 255, 255])}
    },
    'clove_lemon': {
        'W': {'lower': np.array([0, 82, 0]), 'upper': np.array([179, 255, 255])},
        'B': {'lower': np.array([0, 0, 0]), 'upper': np.array([106, 255, 255])}
    },
    'avocado': {
        'W': {'lower': np.array([0, 0, 0]), 'upper': np.array([179, 255, 114])},
        'B': {'lower': np.array([0, 55, 62]), 'upper': np.array([179, 255, 255])}
    },
    'bergamot': {
        'W': {'lower': np.array([0, 63, 0]), 'upper': np.array([179, 255, 255])},
        'B': {'lower': np.array([2, 40, 70]), 'upper': np.array([179, 255, 255])}
    },
    'pear': {
        'W': {'lower': np.array([0, 33, 1]), 'upper': np.array([61, 255, 255])},
        'B': {'lower': np.array([0, 64, 42]), 'upper': np.array([179, 255, 255])}
    },
}

CLASS_ID_MAPPING = {
    '0': "acerola",
    '1': "lemon",
    '2': "cherry_tomato",
    '3': "khaki",
    '4': "banana",
    '5': "lime",
    '6': "clove_lemon",
    '7': "avocado",
    '8': "bergamot",
    '9': "pear"
}

def clean_mask(mask):
    """Final post-processing."""
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean_mask

def segment_by_specific_color(image, fruit_class, background_type):
    if image is None: return None

    if fruit_class not in HSV_RANGES or background_type not in HSV_RANGES[fruit_class]:
        print(f"  -> WARNING: HSV rule for '{fruit_class}' with background '{background_type}' not defined. Returning black mask.")
        return np.zeros(image.shape[:2], dtype="uint8")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    params = HSV_RANGES[fruit_class][background_type]
    
    mask = cv2.inRange(hsv, params['lower'], params['upper'])
    
    if 'lower2' in params:
        mask2 = cv2.inRange(hsv, params['lower2'], params['upper2'])
        mask = cv2.bitwise_or(mask, mask2)
        
    final_mask = clean_mask(mask)
    return final_mask

# Note: The IoU function is not used in this batch script, but is kept for completeness.
def calculate_iou(gt_mask, pred_mask):
    """Calculates the Intersection over Union (IoU) metric between two masks."""
    if gt_mask is None or pred_mask is None: return 0.0
    gt_mask = gt_mask > 127
    pred_mask = pred_mask > 127
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    if np.sum(union) == 0: return 1.0 if np.sum(intersection) == 0 else 0.0
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

if __name__ == '__main__':
    full_dataset_folder = 'dataset_total'
    # Folder where ALL generated masks will be saved
    output_masks_folder = 'masks_geradas_final'

    # Create the output folder if it doesn't exist
    os.makedirs(output_masks_folder, exist_ok=True)

    # Find all images in the input folder
    image_list = glob.glob(os.path.join(full_dataset_folder, '*.png'))
    total_images = len(image_list)

    if total_images == 0:
        print(f"ERROR: No images found in folder '{full_dataset_folder}'.")
        print("Please make sure to place all your images in this folder.")
    else:
        print(f"Found {total_images} images. Starting batch mask generation...")

    # Main loop to generate all masks
    for i, image_path in enumerate(image_list):
        base_name = os.path.basename(image_path)
        print(f"Processing {i+1}/{total_images}: {base_name}")

        class_id = base_name.split('-')[0]
        # Detect the background type ('W' or 'B') from the filename
        background_type = 'W' if '_W.' in base_name else 'B'
        
        # Check if the class ID exists in our mapping
        if class_id not in CLASS_ID_MAPPING:
            print(f"  -> WARNING: Mapping for class ID '{class_id}' not found. Skipping.")
            continue
        
        class_name = CLASS_ID_MAPPING[class_id]
        read_image = cv2.imread(image_path)
        
        # Generate the mask using the calibrated function
        generated_mask = segment_by_specific_color(read_image, class_name, background_type)
        
        if generated_mask is not None:
            # Save the mask in the output folder with the same name as the original image
            output_path = os.path.join(output_masks_folder, base_name)
            cv2.imwrite(output_path, generated_mask)
        else:
            print(f"  -> Failed to process image: {base_name}")
            
    print("\nBatch processing completed!")
    print(f"All masks were saved in '{output_masks_folder}'.")