import cv2
import numpy as np
from collections import namedtuple
import re
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
from pathlib import Path
import os

def save_image_with_modified_filename(image, input_filename, suffix):
    filename, ext = os.path.splitext(input_filename)
    output_filename = filename + suffix + ext
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image)
    print(f"Image saved: {output_path}")

output_folder = "output"

# Define a named tuple for Rectangle
Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])

# Define the Levels class for different OCR levels
class Levels:
    PAGE = 1
    BLOCK = 2
    PARAGRAPH = 3
    LINE = 4
    WORD = 5

def intersect_area(a, b):
    """Calculate intersection area between two rectangles."""
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    return float(dx * dy) if (dx >= 0) and (dy >= 0) else 0.

def normalize_images(images):
    """Convert all images into 3-dimensional images via cv2.COLOR_GRAY2BGR."""
    return [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image for image in images]

def threshold_image(img_src):
    """Grayscale image and apply Otsu's threshold."""
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_thresh, img_gray

def mask_image(img_src, lower, upper):
    """Convert image from RGB to HSV and create a mask for given lower and upper boundaries."""
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(lower, np.uint8)  # Lower HSV value
    hsv_upper = np.array(upper, np.uint8)  # Upper HSV value
    img_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
    return img_mask, img_hsv

def apply_mask(img_src, img_mask):
    """Apply bitwise conjunction of source image and image mask."""
    img_result = cv2.bitwise_and(img_src, img_src, mask=img_mask)
    return img_result

def denoise_image(img_src):
    """Denoise image with a morphological transformation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_denoise = cv2.morphologyEx(img_src, cv2.MORPH_OPEN, kernel, iterations=1)
    return img_denoise

def draw_word_boundings(img_src, data_ocr, highlighted_words=False):
    """Draw word bounding boxes."""
    img_result = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR) if img_src.ndim == 2 else img_src.copy()
    for i in range(len(data_ocr['text'])):
        if data_ocr['level'][i] != Levels.WORD:
            continue
        if highlighted_words and not data_ocr['highlighted'][i]:
            continue
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top'][i], data_ocr['width'][i], data_ocr['height'][i])
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img_result

def draw_contour_boundings(img_src, img_thresh ,img_mask, threshold_area=400):
    """Draw contour bounding and contour bounding box."""
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img_src.copy()
    img_box = img_src.copy()
    os.makedirs("combinedPreprocessing\highlighted_boxes", exist_ok=True)
    for idx, c in enumerate(contours):
        highlighted_box = img_thresh.copy()
        if cv2.contourArea(c) < threshold_area:
            continue
        cv2.drawContours(img_contour, contours, idx, (0, 0, 255), 2, cv2.LINE_4)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)
        cropped_image = highlighted_box[y:y+h, x:x+w]
        # filename, ext = os.path.splitext("cropped")
        # output_filename = filename + ext
        # output_path = os.path.join("highlighted_boxes", output_filename)
        cv2.imwrite(f"combinedPreprocessing\highlighted_boxes\cropped_{idx}.jpg", cropped_image)
        print(f'cropped_{idx}')
    return img_contour, img_box

def find_highlighted_words(img_mask, data_ocr, threshold_percentage=25):
    """Find highlighted words by calculating how much of the word's area contains white pixels compared to black pixels."""
    data_ocr['highlighted'] = [False] * len(data_ocr['text'])
    for i in range(len(data_ocr['text'])):
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top'][i], data_ocr['width'][i], data_ocr['height'][i])
        rect_threshold = (w * h * threshold_percentage) / 100
        img_roi = img_mask[y:y+h, x:x+w]
        count = cv2.countNonZero(img_roi)
        if count > rect_threshold:
            data_ocr['highlighted'][i] = True
    return data_ocr

def save_highlighted(image_path):
    if image_path.exists():
        print("hi")
        img_orig = cv2.imread(str(image_path))
        if img_orig is not None:
            # Grayscale and apply Otsu's threshold
            img_thresh, img_gray = threshold_image(img_orig)
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            # Perform OCR on the thresholded image
            data_ocr = pytesseract.image_to_data(img_thresh, lang='eng', config='--psm 6', output_type=Output.DICT)

            # Print all the text identified
            # for text in data_ocr['text']:
            #     print(text + "\n")

            # Concatenate all the text identified
            all_text = "\n".join(data_ocr['text'])
            
            # Print the concatenated text
            # print(all_text)

            # # Black highlight colour range
            # hsv_lower = [0, 0, 0]
            # hsv_upper = [180, 255, 30]
            
            # # White highlight colour range
            # hsv_lower = [0, 0, 231]
            # hsv_upper = [180, 18, 255]
            
            # # Yellow highlight colour range / not needed
            # hsv_lower = [30, 100, 100]
            # hsv_upper = [35, 255, 255]
            
            # # Yellow highlight colour range
            hsv_lower = [22, 30, 30]
            hsv_upper = [45, 255, 255]
            
            # # Blue highlight colour range
            # hsv_lower = [90, 50, 70] 
            # hsv_upper = [128, 255, 255]  
            
            # # Red1 highlight colour range
            # hsv_lower = [159, 50, 70]
            # hsv_upper = [180, 255, 255]
            
            # # Red2 highlight colour range
            # hsv_lower = [0, 50, 70]
            # hsv_upper = [9, 255, 255]
            
            # # Green highlight colour range
            # hsv_lower = [30, 30, 30]
            # hsv_upper = [120, 255, 255]       

            # # Purple and magenta highlight colour range
            # hsv_lower = [129, 30, 30]
            # hsv_upper = [158, 255, 255]
            
            # # pink highlight colour range
            # hsv_lower = [158, 30, 30]
            # hsv_upper = [179, 255, 255]
            
            # Orange highlight colour range
            # hsv_lower = [10, 50, 70]
            # hsv_upper = [24, 255, 255]
            
            # # Gray highlight colour range
            # hsv_lower = [0, 0, 40]
            # hsv_upper = [180, 18, 230]


            # Color segmentation
            img_mask, img_hsv = mask_image(img_orig, hsv_lower, hsv_upper)
            

            # Noise reduction
            img_mask_denoised = denoise_image(img_mask)

            # Apply mask on original image
            img_orig_masked = apply_mask(img_orig, img_mask=img_mask_denoised)

            # Apply mask on thresholded image
            img_thresh_masked = apply_mask(img_thresh, img_mask=img_mask_denoised)

            # Find highlighted words
            data_ocr = find_highlighted_words(img_mask_denoised, data_ocr, threshold_percentage=25)

            # Draw contour bounding and contour bounding box
            img_orig_bounding_contour, img_orig_bounding_box = draw_contour_boundings(img_orig,img_thresh ,img_mask=img_mask_denoised)

            # Draw word boundings for highlighted words
            # img_thresh_word_boundings = draw_word_boundings(img_thresh, data_ocr, highlighted_words=True)
            # img_orig_word_boundings = draw_word_boundings(img_orig, data_ocr, highlighted_words=True)

            # Display or save your processed images
            # You can use cv2.imshow() or plt.imshow() to display images
            # You can use cv2.imwrite() or plt.savefig() to save images

            # Example: Displaying the original image and the image with highlighted words
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img_orig_bounding_box, cv2.COLOR_BGR2RGB))
            plt.title('Image with Highlighted Words')
            plt.show()
            
            # Save the images with modified filenames
            save_image_with_modified_filename(img_orig_bounding_box, 'yellowHighlight.jpg', "_output")
            # save_image_with_modified_filename(img_orig_word_boundings, image_filename, "_output")
        else:
            print("Error: Failed to load the image.")
    else:
        print("Error: Image file does not exist.")

def save_lines(image_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## (2) threshold
    # Increase the threshold value for less sensitivity
    th, threshed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    ## (3) minAreaRect on the nozeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx,cy), (w,h), ang = ret
    if w < h:
        w, h = h, w
        ang -= 90

    ## (4) Find rotated matrix, do rotation
    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated_gray = cv2.warpAffine(threshed, M, (gray.shape[1], gray.shape[0]))
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    ## (5) find and draw the upper and lower boundary of each line on the rotated_gray image
    hist = cv2.reduce(rotated_gray, 1, cv2.REDUCE_AVG).reshape(-1)

    # Dynamically calculate threshold for better line segmentation
    th = np.mean(hist) * 0.9  # Adjust the multiplier as needed

    H, W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]

    # Merge adjacent lines to avoid splitting text within the same line
    merged_uppers = [uppers[0]]
    merged_lowers = []
    for i in range(1, len(uppers)):
        if uppers[i] - merged_uppers[-1] < 10:  # Adjust the threshold as needed
            continue
        merged_uppers.append(uppers[i])
        merged_lowers.append(lowers[i-1])

    # Create folders if they don't exist
    os.makedirs("results_gray", exist_ok=True)
    os.makedirs("results_colour", exist_ok=True)

    # Filter out lines that appear to be empty based on the percentage of white pixels
    filtered_uppers = []
    filtered_lowers = []
    for upper, lower in zip(merged_uppers, merged_lowers):
        line_region = rotated_gray[upper:lower, :]
        if line_region.size > 0:
            white_pixel_percentage = np.sum(line_region == 255) / (line_region.shape[0] * line_region.shape[1])
            if white_pixel_percentage < 0.95:  # Adjust the threshold as needed
                filtered_uppers.append(upper)
                filtered_lowers.append(lower)

    # Process the cropped regions and save
    for i, (y1, y2) in enumerate(zip(filtered_uppers, filtered_lowers)):
        y1 = max(0, y1 - 5)
        y2 = min(H, y2 + 5)
        
        if y2 - y1 > 5:
            # Crop the region based on the current y1 and y2 from the grayscale image
            cropped_gray = rotated_gray[y1:y2, 0:W]
            inverted_gray = cv2.bitwise_not(cropped_gray)
            
            # Crop the region based on the current y1 and y2 from the colored image
            cropped_img = rotated_img[y1:y2, 0:W]
            
            # Write the cropped images to files in the respective folders
            cv2.imwrite(f"combinedPreprocessing/results_gray/result_gray_{i}.png", inverted_gray)
            cv2.imwrite(f"combinedPreprocessing/results_colour/result_color_{i}.png", cropped_img)
            print(f"Wrote result_gray_{i}.png and result_color_{i}.png")

    # Draw lines on the rotated_img
    rotated_with_lines = rotated_gray.copy()
    for y in filtered_uppers:
        cv2.line(rotated_with_lines, (0, y), (W, y), (255, 0, 0), 1)

    for y in filtered_lowers:
        cv2.line(rotated_with_lines, (0, y), (W, y), (0, 255, 0), 1)

    # Save the image with lines
    cv2.imwrite(f"result.png", rotated_with_lines)

image_path =Path('test5.png')
save_highlighted(image_path)
save_lines(image_path)