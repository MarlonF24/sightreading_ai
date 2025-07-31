from pathlib import Path

def enhance_pdf_resolution(input_pdf: Path, output_pdf: Path, min_dpi: int = 300, conversion_dpi: int = 150):
    """
    Check PDF resolution and enhance it if below minimum DPI threshold.
    
    :param input_pdf: Path to the input PDF file
    :param output_pdf: Path to save the enhanced PDF file
    :param min_dpi: Minimum DPI threshold (default: 300)
    :param conversion_dpi: Initial DPI for PDF to image conversion (default: 150)
    :return: True if enhancement was applied, False if not needed
    """
    
    from pdf2image import convert_from_path
    from PIL import Image

    # Convert PDF to images at the conversion DPI
    pages = convert_from_path(
        str(input_pdf),
        dpi=conversion_dpi,
        poppler_path="C:\\Users\\marlo\\anaconda3\\pkgs\\poppler-24.09.0-h6558a74_1\\Library\\bin\\"
    )
    
    enhanced_pages = []
    enhancement_applied = False
    
    for img in pages:
        # Get the current DPI from the image info
        current_dpi = img.info.get('dpi', (conversion_dpi, conversion_dpi))
        if isinstance(current_dpi, tuple):
            current_dpi = current_dpi[0]  # Use x_dpi
        
        print(f"Current DPI: {current_dpi}")
        
        if current_dpi < min_dpi:
            # Calculate scale factor needed to reach minimum DPI
            scale_factor = min_dpi / current_dpi
            width, height = img.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            print(f"Scaling image from {width}x{height} to {new_width}x{new_height}")
            print(f"DPI will increase from {current_dpi} to {min_dpi}")
            
            # Upscale the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            enhancement_applied = True
        else:
            print(f"Resolution {current_dpi} DPI meets minimum requirement of {min_dpi} DPI")
        
        # Convert to black and white for smaller file size (optional)
        img_bw = img.convert("1")
        enhanced_pages.append(img_bw)
    
    # Save enhanced PDF
    if enhanced_pages:
        # Set the DPI for the output
        output_dpi = min_dpi if enhancement_applied else current_dpi
        
        enhanced_pages[0].save(
            str(output_pdf), 
            save_all=True, 
            append_images=enhanced_pages[1:],
            format='PDF',
            resolution=output_dpi,
            dpi=(output_dpi, output_dpi)
        )
        print(f"Saved enhanced PDF to {output_pdf}")
    else:
        raise ValueError("No pages were processed from the PDF")
    
    return enhancement_applied



# Example usage
if __name__ == "__main__":
    # input_pdf = Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/pdf_in/p-harris-improve-your-sight-reading-3pdf_compress.pdf")
    # output_pdf = Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/pdf_in/output.pdf")

    # enhance_pdf_resolution(input_pdf, output_pdf, min_dpi=1000)
    pass


