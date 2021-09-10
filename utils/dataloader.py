import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

	
def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".PNG", ".JPG"])

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

class pair_loader(data.Dataset):

	def __init__(self, images_path, patch_size=128):
		
		high_light_dir = 'high'
		low_light_dir = 'low'
		low_files = sorted(os.listdir(os.path.join(images_path, low_light_dir)))
		high_files = sorted(os.listdir(os.path.join(images_path, high_light_dir)))

		self.low_files = [os.path.join(images_path, low_light_dir, x) for x in low_files if is_img_file(x)]
		self.high_files = [os.path.join(images_path, high_light_dir, x) for x in high_files if is_img_file(x)]

		self.size =patch_size

		self.img_num = len(self.low_files)
		print("Total training examples:", self.img_num)

	def __len__(self):
		return self.img_num
		

	def __getitem__(self, index):
		tar_index = index % self.img_num
		low = torch.from_numpy(np.float32(load_img(self.low_files[tar_index])))
		high = torch.from_numpy(np.float32(load_img(self.high_files[tar_index])))

		low = low.permute(2,0,1)
		high = high.permute(2,0,1)

		low_filename = os.path.split(self.low_files[tar_index])[-1]
		high_filename = os.path.split(self.high_files[tar_index])[-1]

		#Crop Input and Target
		ps = self.size
		H = low.shape[1]
		W = low.shape[2]
		# r = np.random.randint(0, H - ps) if not H-ps else 0
		# c = np.random.randint(0, W - ps) if not H-ps else 0
		if H-ps==0:
			r=0
			c=0
		else:
			r = np.random.randint(0, H - ps)
			c = np.random.randint(0, W - ps)
		low = low[:, r:r + ps, c:c + ps]
		high = high[:, r:r + ps, c:c + ps]

		return low, high, low_filename, high_filename

