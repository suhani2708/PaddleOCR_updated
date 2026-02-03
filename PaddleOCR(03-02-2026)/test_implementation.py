"""
Test script to run the PaddleOCR Excel export functionality.
"""

import os
from paddle_ocr_excel_exporter import PaddleOCRExcelExporter

def test_basic_ocr_export():
    """Test basic OCR to Excel export functionality."""
    print("Testing basic OCR to Excel export...")
    
    # Initialize the exporter
    exporter = PaddleOCRExcelExporter(
        lang='en',
        confidence_threshold=0.8,  # Set a reasonable threshold
        enable_orientation=True,
        y_tolerance=15,  # Increase tolerance for row detection
        reading_direction='ltr'
    )
    
    # Set up paths
    image_path = "docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
    excel_path = "test_output.xlsx"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        # Try alternative path
        image_path = "c:/Users/91909/Desktop/Paddleocr(02.02.2026)/PaddleOCR(updated)/docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
        if not os.path.exists(image_path):
            print("Image still not found at alternative path")
            return False
    
    print(f"Using image: {image_path}")
    print(f"Exporting to: {excel_path}")
    
    try:
        # Export to Excel
        result_path = exporter.export_to_excel(image_path, excel_path)
        print(f"✅ Successfully exported to: {result_path}")
        
        # Generate quality report
        report = exporter.generate_report(image_path)
        print("\n📊 Quality Report:")
        print(report)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during export: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_templates():
    """Test OCR export with custom templates."""
    print("\nTesting OCR export with custom templates...")

    # Initialize the exporter
    exporter = PaddleOCRExcelExporter(
        lang='en',
        confidence_threshold=0.8,
        enable_orientation=True
    )

    # Create custom templates
    from paddle_ocr_excel_exporter import DEFAULT_TEMPLATES

    for template_name, config in DEFAULT_TEMPLATES.items():
        exporter.create_custom_template(template_name, config)
        print(f"✅ Created template: {template_name}")

    # Set up paths
    image_path = "docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
    excel_path = "test_output_template.xlsx"

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False

    try:
        # Export with standard template
        result_path = exporter.export_with_template(image_path, excel_path, 'standard')
        print(f"✅ Successfully exported with template to: {result_path}")
        return True

    except Exception as e:
        print(f"❌ Error during template export: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics():
    """Test quality metrics functionality."""
    print("\nTesting quality metrics...")
    
    # Initialize the exporter
    exporter = PaddleOCRExcelExporter(
        lang='en',
        confidence_threshold=0.8,
        enable_orientation=True
    )
    
    # Set up path
    image_path = "docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    try:
        # Get quality metrics
        metrics = exporter.get_quality_metrics(image_path)
        print("✅ Quality metrics retrieved:")
        print(f"  Total rows: {metrics['total_rows']}")
        print(f"  Total words: {metrics['total_words']}")
        print(f"  Avg confidence: {metrics['avg_confidence']:.2f}")
        print(f"  Low confidence words: {metrics['low_confidence_words']} ({metrics['low_confidence_percentage']:.1f}%)")
        print(f"  Confidence distribution: {metrics['confidence_distribution']}")
        return True
        
    except Exception as e:
        print(f"❌ Error getting quality metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting PaddleOCR Excel Export Test")
    print("="*50)
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_basic_ocr_export():
        success_count += 1
    
    if test_with_templates():
        success_count += 1
        
    if test_quality_metrics():
        success_count += 1
    
    print("\n" + "="*50)
    print(f"Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("All tests passed! The implementation is working correctly.")
    else:
        print(f"{total_tests - success_count} test(s) failed.")

    print("\nGenerated files:")
    print("- test_output.xlsx: Basic OCR export")
    print("- test_output_template.xlsx: Template-based export")