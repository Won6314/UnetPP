from set1.util.iomanager import imreads, imwrites
from albumentations import RandomRotate90, Flip, Transpose, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,Blur,\
	ShiftScaleRotate, OneOf, OpticalDistortion, Compose, GridDistortion, IAAPiecewiseAffine, IAASharpen, CLAHE, IAAEmboss, RandomContrast,\
	RandomBrightness


def strong_aug(p=1):
	return Compose([
		RandomRotate90(),
		Flip(),
		Transpose(),
		OneOf([
			IAAAdditiveGaussianNoise(),
			GaussNoise(),
		], p=0.2),
		OneOf([
			MotionBlur(p=.2),
			MedianBlur(blur_limit=3, p=.1),
			Blur(blur_limit=3, p=.1),
		], p=0.2),
		ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
		OneOf([
			OpticalDistortion(p=0.3),
			GridDistortion(p=.1),
			IAAPiecewiseAffine(p=0.3),
		], p=0.2),
		OneOf([
			CLAHE(clip_limit=2),
			IAASharpen(),
			IAAEmboss(),
			RandomContrast(),
			RandomBrightness(),
		], p=0.3),
		# HueSaturationValue(p=0.3),
	], p=p)
# https://www.kaggle.com/alexanderliao/image-augmentation-demo-with-albumentation


if __name__ == '__main__':
	train_input = imreads('./data/stage1_train_neat', pattern='*input.png', same=False)
	train_label = imreads('./data/stage1_train_neat', pattern='*label.png')