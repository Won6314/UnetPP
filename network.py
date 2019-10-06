import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T


class Convblock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		m = []
		m.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
		m.append(nn.InstanceNorm2d(out_ch))
		m.append(nn.ReLU(True))
		m.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
		m.append(nn.InstanceNorm2d(out_ch))
		m.append(nn.ReLU(True))
		self.layer = nn.Sequential(*m)

	def forward(self, input):
		return self.layer(input)


class UnetPP(nn.Module):
	def __init__(self, in_ch=3, out_ch=1):
		super().__init__()
		ch = [32, 64, 128, 256, 512]

		self.downsample = nn.MaxPool2d(2,2)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
		self.conv00 = Convblock(in_ch, ch[0])
		self.conv01 = Convblock(ch[0]+ch[1], ch[0])
		self.conv02 = Convblock(ch[0]*2 + ch[1], ch[0])
		self.conv03 = Convblock(ch[0]*3 + ch[1], ch[0])
		self.conv04 = Convblock(ch[0]*4 + ch[1], ch[0])

		self.conv10 = Convblock(ch[0], ch[1])
		self.conv11 = Convblock(ch[1] + ch[2], ch[1])
		self.conv12 = Convblock(ch[1]*2 + ch[2], ch[1])
		self.conv13 = Convblock(ch[1]*3 + ch[2], ch[1])

		self.conv20 = Convblock(ch[1], ch[2])
		self.conv21 = Convblock(ch[2] + ch[3], ch[2])
		self.conv22 = Convblock(ch[2]*2 + ch[3], ch[2])

		self.conv30 = Convblock(ch[2], ch[3])
		self.conv31 = Convblock(ch[3] + ch[4], ch[3])

		self.conv40 = Convblock(ch[3], ch[4])

		self.conv_end = nn.Conv2d(ch[0], out_ch, 1, 1, 0)

	def forward(self, input):
		out00 = self.conv00(input)
		out10 = self.conv10(self.downsample(out00))
		out20 = self.conv20(self.downsample(out10))
		out30 = self.conv30(self.downsample(out20))
		out40 = self.conv40(self.downsample(out30))

		out01 = self.conv01(torch.cat([out00, self.upsample(out10)], dim=1))
		out11 = self.conv11(torch.cat([out10, self.upsample(out20)], dim=1))
		out21 = self.conv21(torch.cat([out20, self.upsample(out30)], dim=1))
		out31 = self.conv31(torch.cat([out30, self.upsample(out40)], dim=1))

		out02 = self.conv02(torch.cat([out00, out01, self.upsample(out11)], dim=1))
		out12 = self.conv12(torch.cat([out10, out11, self.upsample(out21)], dim=1))
		out22 = self.conv22(torch.cat([out20, out21, self.upsample(out31)], dim=1))

		out03 = self.conv03(torch.cat([out00, out01, out02, self.upsample(out12)], dim=1))
		out13 = self.conv13(torch.cat([out10, out11, out12, self.upsample(out22)], dim=1))

		out04 = self.conv04(torch.cat([out00, out01, out02, out03, self.upsample(out13)], dim=1))

		outs = [self.conv_end(out) for out in [out01, out02, out03, out04]]

		return outs
