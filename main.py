import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
from logger import set_logging_path, logger
from DataLoader import Dataset
import argparse
from os.path import join
from datetime import datetime
from network import UnetPP
from loss import LosswithLogit

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', '-bs', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('-lr', '--learning_rate', type=float, default = 1e-4)

args = parser.parse_args()

set_logging_path(path=join(args.log_dir, 'log{}.txt'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))))

trainloader = data.DataLoader(dataset=Dataset(root="./data/stage1_train_neat"),
							 batch_size=args.batch_size,
							 shuffle=True)
testloader = data.DataLoader(dataset=Dataset(root="./data/stage1_test_neat"),
							 batch_size=args.batch_size,
							 shuffle=False)

device = torch.device("cuda:0")
model = UnetPP()
model.to(device)

criterion = LosswithLogit()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=, gamma=)

from set1.util.imshow import nshow
for epoch in range(10):
	for i, (images, labels) in enumerate(trainloader):
		images = images.cuda()
		labels = labels.cuda()

		outputs = model(images)
		loss = criterion(outputs[3], labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i) % 100 == 0:
			logger.info("Epoch [{}], Step [{}] Loss: {:.4f}"
				  .format(epoch + 1, i, loss.item()))

