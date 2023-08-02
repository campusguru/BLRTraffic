import os
from utils.data_utils import get_file_type
from recognize import detect_on_image, detect_on_video

OUTPUT_FOLDER_PATH = './output'

def run(input_path):
    input_file_type = get_file_type(input_path)
    if input_file_type == "image":
        print('Detected image...')
        detect_on_image.run_on_image(input_path)
    elif input_file_type == "video":
        print('Detected video...')
        output_path = os.path.join(OUTPUT_FOLDER_PATH, os.path.basename(input_path))
        detect_on_video.get_plates_from_video(input_path, output_path)
    else:
        print(f"Provided file format {input_file_type} not supported, Please recheck the given path {input_path}")

if __name__ == "__main__":
    input_file_path = "/Users/raj/Downloads/Projects/BLRTraffic/test/test1.jpg"
    run(input_file_path)
