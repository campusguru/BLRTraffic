import cv2
import numpy as np

def crop(img, coords):
    cropped = img[coords[1]:coords[3], coords[0]:coords[2]]
    return cropped

def preprocess_image(img):
    normalize = cv2.normalize(img, np.zeros((img.shape[0], img.shape[1])), 0, 255, cv2.NORM_MINMAX)
    denoise = cv2.fastNlMeansDenoisingColored(normalize, h=10, hColor=10, templateWindowSize=7, searchWindowSize=15)
    grayscale = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return threshold

def plot_images(images, width, height, columns=2, rows=3):
    fig = plt.figure(figsize=(width, height))
    for i, file in enumerate(images):
        img = plt.imread(file)
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
    plt.show()

def plot_box(img, coords, label=None, color=[0, 150, 255], line_thickness=3):
    # Plots box on image
    c1, c2 = (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    # Plots label on image, if exists
    if label:
        tf = max(line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)