import pypdfium2 as pdfium
from pathlib import Path
from src.extraction import LlamaVisionESGExtractor
from src.vnd_detector import VisualNumericalDissonanceDetector

if __name__ == "__main__":
    extractor = LlamaVisionESGExtractor()
    extractor.load_model()
    detector = VisualNumericalDissonanceDetector(extractor_instance=extractor)

    # Test against a known page containing an emissions chart
    sample_pdf = list(Path("./data/raw_pdfs").glob("*Google*2023*.pdf"))[0]
    pdf = pdfium.PdfDocument(str(sample_pdf))
    
    # Check page 10 (or any chart page)
    test_page_idx = 9
    img = pdf[test_page_idx].render(scale=0.8).to_pil().convert("RGB")
    pdf.close()

    result = detector.analyze_page_chart(
        image=img, 
        company="Google", 
        fiscal_year=2023, 
        page_number=test_page_idx + 1
    )

    print("\n" + "="*25 + " VND TEST RESULT " + "="*25)
    print(f"Company:           {result.company} ({result.fiscal_year})")
    print(f"Page Number:       {result.page_number}")
    print(f"Y-Axis Origin:     {result.y_axis_origin}")
    print(f"Dissonance Factor: {result.dissonance_factor}x")
    print(f"Distortion Flag:   {result.is_distorted}")
    print(f"Diagnostic Note:   {result.explanation}")
    print("="*67)

    extractor.unload_model()