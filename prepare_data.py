from utils.data_utils import download_data, extract_files, get_files_list, split_train_test
# from recognize.get_license_num import plot_sample_images

data_folder = "/Users/raj/Downloads/Projects/Datasets/"
dataset_name = 'anpr-dataset'

if __name__ == "__main__":
	download_data(dataset_name)
	extract_files(data_folder)
	image_file_list = get_files_list(data_folder)
	print(image_file_list)
	# plot_sample_images(k=5)
	split_train_test(data_folder, dataset=dataset_name)
