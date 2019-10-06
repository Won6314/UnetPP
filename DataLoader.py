import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from glob import glob
from os.path import join
from PIL import Image


class ExpandTransform:
	def __call__(self, img:torch.Tensor):
		if img.shape[0] == 4:
			return img[:3, :, :]
		return img

transform = T.Compose([
	T.Resize(size=[256, 256]),
	T.ToTensor(),
	ExpandTransform()])
class Dataset(data.Dataset):
	def __init__(self, root):
		self.input_list = sorted(glob(join(root, '*input.png')))
		self.label_list = sorted(glob(join(root, '*label.png')))
		assert self.input_list.__len__() == self.label_list.__len__()


	def __getitem__(self, index):
		return transform(Image.open(self.input_list[index])), \
			   transform(Image.open(self.label_list[index]))


	def __len__(self):
		return self.input_list.__len__()


