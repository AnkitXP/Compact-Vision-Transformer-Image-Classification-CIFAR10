import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize

def configure():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, help="train, test or predict", required=True)
	parser.add_argument("--load", type=str, help="model checkpoint load path", default='cifar10-classifier.ckpt')
	return parser.parse_args()

def main(args, model_configs):

	model = MyModel(model_configs)

	if args.mode == 'train':
		
		# checkpoint_path = os.path.join(model_configs.save_dir, args.load) 
		# model.load(checkpoint_path)

		x_train, y_train, x_test, y_test = load_data(model_configs.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		model.evaluate(x_test, y_test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		
		checkpoint_path = os.path.join(model_configs.save_dir, args.load) 
		model.load(checkpoint_path)

		_, _, x_test, y_test = load_data(model_configs.data_dir)
		model.evaluate(x_test, y_test)
		

	elif args.mode == 'predict':

		checkpoint_path = os.path.join(model_configs.save_dir, args.load) 
		model.load(checkpoint_path)

		# Loading private testing dataset
		x_test = load_testing_images(model_configs.data_dir)

		# visualizing the first testing image to check your image shape
		visualize(x_test[0], model_configs.result_dir + 'test.png')

		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test)
		np.save(model_configs.result_dir + 'predictions.npy', predictions)

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	args = configure()
	main(args, model_configs)
