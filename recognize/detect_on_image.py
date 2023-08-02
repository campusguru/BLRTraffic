import cv2
import matplotlib.pyplot as plt
from recognize.get_license_num import get_plates


def run_on_image(image_path):
    # image_path = './anpr-dataset/test/images/CG12.jpg'
    test_image = cv2.imread(image_path)
    detected_image, det_predictions, ocr_predictions = get_plates(test_image)
    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.show()
    print(det_predictions, ocr_predictions)
