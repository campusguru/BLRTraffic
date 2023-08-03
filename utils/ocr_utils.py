import easyocr
# from paddleocr import PaddleOCR
from utils.img_utils import preprocess_image, crop
from utils.text_utils import clean_text


# ocr_reader = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
ocr_reader = easyocr.Reader(['en'], gpu=True)

def ocr_plate(src):
    # Preprocess the image for better OCR results
    preprocessed = preprocess_image(src)

    # OCR the preprocessed image
    results = ocr_reader.ocr(preprocessed, det=False, cls=True)

    # Get the best OCR result
    plate_text, ocr_confidence = max(
        results,
        key=lambda ocr_prediction: max(
            ocr_prediction,
            key=lambda ocr_prediction_result: ocr_prediction_result[1],
        ),
    )[0]

    # Filter out anything but uppercase letters, digits, hypens and whitespace.
    # Also, remove hypens and whitespaces at the first and last positions
    plate_text_filtered = clean_text(plate_text)

    return {"plate": plate_text_filtered, "ocr_conf": ocr_confidence}

def ocr_plates(src, det_predictions):
    results = []

    for det_prediction in det_predictions:
        plate_region = crop(src, det_prediction['coords'])
        ocr_prediction = ocr_plate(plate_region)
        results.append(ocr_prediction)

    return results
