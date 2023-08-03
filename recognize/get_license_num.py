from utils.img_utils import plot_images, plot_box
from utils.object_detect_utils import detect_plates
from utils.ocr_utils import ocr_plates
from glob import glob


def plot_sample_images(k=5):
	dataset_sample = glob('anpr-dataset/labeled_images/images/G*.jpg')[:k]
	plot_images(dataset_sample, width=16, height=8, columns=5, rows=1)


def get_plates(src):
    print("detecting number plates...")
    det_predictions = detect_plates(src)
    print("Done")
    print("Running OCR on plate...")
    ocr_predictions = ocr_plates(src, det_predictions)
    print("Done")
    print("Displaying Detections on Frame")
    for det_prediction, ocr_prediction in zip(det_predictions, ocr_predictions):
        plot_box(src, det_prediction['coords'], ocr_prediction['plate'])
    print("Done")
    return src, det_predictions, ocr_predictions

if __name__ == "__main__":
     plot_sample_images(k=5)

