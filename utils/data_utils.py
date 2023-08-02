import os
import tarfile
import splitfolders
import pandas as pd
from glob import glob
from internetarchive import download

img_exts = (".tif", ".tiff", ".jpg", ".jpeg", ".gif", ".png", ".eps", 
        ".raw", ".cr2", ".nef", ".orf", ".sr2", ".bmp", ".ppm", ".heif")
vid_exts = (".flv", ".avi", ".mp4", ".3gp", ".mov", ".webm", ".ogg", ".qt", ".avchd")
aud_exts = (".flac", ".mp3", ".wav", ".wma", ".aac")
media_ext_dict = {"image": img_exts, "video": vid_exts, "audio": aud_exts}

def get_file_type(input_file_path):
	file_ext = os.path.basename(input_file_path).split('.')[-1]
	for this_type in media_ext_dict:
		if file_ext in media_ext_dict[this_type]:
			return this_type
	return 'unknown'


def download_data(dataset='anpr-dataset'):
    if dataset=='anpr-dataset':
        download("anpr-dataset", files=["anpr-dataset.tar.gz"], verbose=True, no_directory=True) # type: ignore
    else:
        raise Exception(f'Unknown dataset {dataset}. Please check the name')


def get_files_list(input_folder):
	dataset_files = glob(input_folder + "/" + 'anpr-dataset/labeled_images/images/*.jpg')
	return pd.DataFrame(data={'File': dataset_files})


def extract_files(input_folder):
	with tarfile.open(os.path.join(input_folder, "anpr-dataset.tar.gz")) as tar:
    		tar.extractall(input_folder)


def split_train_test(input_folder, dataset, split_ratio=(0.8, 0.1, 0.1)):
	splitfolders.ratio(os.path.join(input_folder, f'{dataset}/labeled_images'), output=os.path.join(input_folder, f'{dataset}/{dataset}_split'), seed=420, ratio=split_ratio)
