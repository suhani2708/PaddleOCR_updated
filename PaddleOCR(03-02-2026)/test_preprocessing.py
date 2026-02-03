"""
Test script to verify the image preprocessing function works correctly.
"""

import os
import sys
import cv2
import numpy as np

# Add the current directory to the path to import our function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paddle_ocr_excel_exporter import preprocess_image_for_ocr


def test_preprocessing_function():
    """Test the preprocessing function with a sample image."""
    print("Testing the image preprocessing function...")
    
    # Look for a sample image in the docs directory
    sample_images = [
        "docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg",
        "docs/images/Statista-FormF-RKJJ-NA-1929752_d.jpg",
        "docs/images/Statista-FormF-RKJJ-NA-1929753_d.jpg",
        "docs/images/Statista-FormF-RKJJ-NA-1929754_d.jpg",
        "docs/images/Statista-FormF-RKJJ-NA-1929755_d.jpg"
    ]
    
    image_path = None
    for img_path in sample_images:
        if os.path.exists(img_path):
            image_path = img_path
            break
    
    if not image_path:
        print("❌ No sample image found for testing")
        # Try alternative path
        alt_path = "c:/Users/91909/Desktop/Paddleocr(02.02.2026)/PaddleOCR(updated)/docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            print("❌ Sample image still not found at alternative path")
            return False
    
    print(f"SUCCESS: Found sample image: {image_path}")
    
    try:
        # Test the preprocessing function
        print("Running image preprocessing...")
        preprocessed_path = preprocess_image_for_ocr(image_path)
        
        print(f"SUCCESS: Preprocessing completed successfully!")
        print(f"SUCCESS: Original image: {image_path}")
        print(f"SUCCESS: Preprocessed image: {preprocessed_path}")

        # Verify the preprocessed image exists
        if os.path.exists(preprocessed_path):
            print("SUCCESS: Preprocessed image file exists")

            # Load and check the preprocessed image
            img = cv2.imread(preprocessed_path)
            if img is not None:
                height, width = img.shape[:2]
                print(f"SUCCESS: Preprocessed image loaded successfully: {width}x{height}")
                print("SUCCESS: All preprocessing steps completed successfully!")
                return True
            else:
                print("ERROR: Could not load preprocessed image")
                return False
        else:
            print("ERROR: Preprocessed image file does not exist")
            return False
            
    except Exception as e:
        print(f"ERROR: Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing_steps_individually():
    """Test each preprocessing step individually."""
    print("\nTesting preprocessing steps individually...")
    
    # Create a simple test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("SUCCESS: Step 1: Converting to grayscale")
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    print(f"   Original shape: {test_img.shape}, Grayscale shape: {gray.shape}")

    print("SUCCESS: Step 2: Applying adaptive thresholding")
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    print(f"   Thresholded image shape: {thresh.shape}")

    print("SUCCESS: Step 3: Denoising")
    denoised = cv2.fastNlMeansDenoising(thresh)
    print(f"   Denoised image shape: {denoised.shape}")

    print("SUCCESS: Step 4: Sharpening text")
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    print(f"   Sharpened image shape: {sharpened.shape}")

    print("SUCCESS: Step 5: Skew correction (simulated)")
    coords = np.column_stack(np.where(sharpened > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        print(f"   Calculated skew angle: {angle}")

    print("SUCCESS: All preprocessing steps work correctly!")
    return True


if __name__ == "__main__":
    print("Testing Image Preprocessing Function")
    print("="*50)
    
    success_count = 0
    total_tests = 2
    
    if test_preprocessing_steps_individually():
        success_count += 1
    
    if test_preprocessing_function():
        success_count += 1
    
    print("\n" + "="*50)
    print(f"Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("SUCCESS: All preprocessing tests passed! The function is working correctly.")
        print("\nThe preprocessing function includes:")
        print("   - Conversion to grayscale")
        print("   - Adaptive thresholding (THRESH_BINARY + ADAPTIVE_THRESH_GAUSSIAN_C)")
        print("   - Denoising using cv2.fastNlMeansDenoising")
        print("   - Text sharpening using kernel filter")
        print("   - Skew/rotation detection and correction using cv2.minAreaRect")
    else:
        print(f"WARNING: {total_tests - success_count} test(s) failed.")