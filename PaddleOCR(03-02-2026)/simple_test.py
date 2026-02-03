"""
Simple test to verify the basic functionality works.
"""

import os
from paddleocr import PaddleOCR

def test_simple_ocr():
    """Test basic OCR functionality."""
    print("Testing basic OCR functionality...")
    
    try:
        # Initialize PaddleOCR with minimal parameters
        ocr = PaddleOCR(
            lang='en',
            use_angle_cls=False,
            return_word_box=True
        )
        
        print("✅ PaddleOCR initialized successfully")
        
        # Test with a simple image
        image_path = "docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            # Try alternative path
            image_path = "c:/Users/91909/Desktop/Paddleocr(02.02.2026)/PaddleOCR(updated)/docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
            if not os.path.exists(image_path):
                print("❌ Image still not found at alternative path")
                return False
        
        print(f"Processing image: {image_path}")
        
        # Run OCR
        result = ocr.ocr(image_path, cls=True)

        print("SUCCESS: OCR completed successfully")
        print(f"Result type: {type(result)}")

        if result:
            print(f"Number of pages in result: {len(result)}")
            if len(result) > 0 and result[0]:
                print(f"Number of text regions detected: {len(result[0])}")

                # Print first few results
                for i, item in enumerate(result[0][:3]):  # Show first 3 items
                    if len(item) >= 2:
                        text_info = item[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            print(f"  {i+1}. Text: '{text}', Confidence: {confidence:.2f}")

        return True

    except Exception as e:
        print(f"ERROR: Error during OCR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test if our modules can be imported."""
    print("\nTesting module imports...")
    
    try:
        from confidence_colors import get_confidence_color
        print("SUCCESS: confidence_colors module imported successfully")

        # Test the function
        color = get_confidence_color(0.5)
        print(f"SUCCESS: get_confidence_color(0.5) = {color}")

        from text_row_detector import detect_text_rows_from_bounding_boxes
        print("SUCCESS: text_row_detector module imported successfully")

        from ocr_utils import extract_word_data
        print("SUCCESS: ocr_utils module imported successfully")

        return True

    except ImportError as e:
        print(f"ERROR: Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting PaddleOCR Simple Test")
    print("="*40)
    
    success_count = 0
    total_tests = 2
    
    if test_imports():
        success_count += 1
    
    if test_simple_ocr():
        success_count += 1
    
    print("\n" + "="*40)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("All tests passed! The basic functionality is working.")
    else:
        print(f"{total_tests - success_count} test(s) failed.")