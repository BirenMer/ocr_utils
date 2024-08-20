import cv2
import numpy as np

def remove_white_space_area(image_path, output_path, kernel_size=10, offset=False, offsetMeasure=30):
    """
    Crops the image to the largest bounding box around non-white areas, optionally applying an offset.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        kernel_size (int): Size of the kernel used for morphological operations to expand areas.
        offset (bool): Whether to apply an offset to the bounding box.
        offsetMeasure (int): Amount of offset to apply if offset is True.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define white color range in grayscale
    white_min = 240  # This can be adjusted based on the whiteness threshold
    white_max = 255
    
    # Create a binary mask where non-white areas are 1
    _, binary_mask = cv2.threshold(gray, white_min, white_max, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to expand the non-white areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    expanded_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the expanded binary mask
    contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No non-white areas found in {image_path}.")
        return
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Apply offset if specified
    if offset:
        x -= offsetMeasure
        y -= offsetMeasure
        w += 2 * offsetMeasure
        h += 2 * offsetMeasure
    
    # Ensure the bounding box coordinates are within image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    # Crop the image based on the adjusted bounding box
    cropped_image = image[y:y + h, x:x + w]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")
    return cropped_image

# Example usage:
# remove_white_space_area(image_path,
#                         output_path,
#                         kernel_size=150,  # Recommeded size for kernel
#                         offset=True,      # Offset to make sure we don't crop any important content.
#                         offsetMeasure=30) # Size of Offset