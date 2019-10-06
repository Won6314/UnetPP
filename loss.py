import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T

class DicewithLogitLoss(nn.Module):
	def __init__(self, eps=1e-6):
		super().__init__()
		self.eps = eps

	def forward(self, out, target):
		out = F.sigmoid(out)
		return 1 - 2 * (((out*target).sum([1,2,3])) / (out.sum([1,2,3]) * target.sum([1,2,3]) + self.eps)).mean()

class LosswithLogit(nn.Module):
	def __init__(self, eps=1e-6):
		super().__init__()
		self.dice = DicewithLogitLoss(eps)
		self.bcewithlogit = nn.BCEWithLogitsLoss()

	def forward(self, out, target):
		return self.bcewithlogit(out, target) + self.dice(out, target)