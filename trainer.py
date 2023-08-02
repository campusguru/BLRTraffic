import os
from ultralytics import YOLO
from utils.img_utils import plot_images

DATASET_FOLDER = '/Users/raj/Downloads/Projects/Datasets'
EPOCHS = 50
BATCH_SIZE = 8
IMAGE_RESOLUTION = 640

def train_model(yaml_path):
# Load the pretrained model
    model = YOLO('./model/yolov8x.pt')
    # Training the model
    return model.train(
    data=yaml_path,
    imgsz=IMAGE_RESOLUTION,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    name='yolov8x_anpr'
    )


def show_training_results(model_path):
    
    curves = [f'{model_path}/F1_curve.png', f'{model_path}/PR_curve.png', f'{model_path}/P_curve.png', f'{model_path}/R_curve.png', f'{model_path}/confusion_matrix.png', f'{model_path}/results.png']
    plot_images(curves, width=30, height=30, columns=2, rows=3)

    with open(f'{model_path}/results.csv') as f:
        file_contents = f.read()
        print(file_contents)

    test_batch = [f'{model_path}/val_batch0_pred.jpg', f'{model_path}/val_batch1_pred.jpg', f'{model_path}/val_batch2_pred.jpg']
    plot_images(test_batch, width=30, height=30, columns=3, rows=1)

if __name__ == "__main__":
    model_path = './runs/detect/anpr_model2'
    train_model(os.path.join(DATASET_FOLDER, 'anpr-dataset/data.yaml'))
    show_training_results(model_path)
