import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

sys.path.insert(0, '../')
from utils import detectors

import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


class custom_dataset(Dataset):
	"""cloth dataset.
	X : image
	Y : params
	"""

	def __init__(self, root_dir, transform=None, index=None):
		self.root_dir = root_dir
		self.transform = transform
		self.face_detector = detectors.FAN()
		self.scale = 1.25
		self.resolution_inp = 224

		self.images = os.listdir(os.path.join(root_dir + 'images'))
		self.images.sort()
		landmark_file = os.path.join(root_dir, 'landmark_data.json')
		with open(landmark_file, 'r') as f:
			self.landmark_data = json.load(f)
			
		if index is not None:
			self.images = self.images[index[0]:index[1]]
			# self.images = images[11:12]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		#get image _height and width
		img_path = os.path.join(self.root_dir + 'images/', self.images[idx])
		og_h, og_w, _ = cv2.imread(img_path).shape

		
		#get label (landmarks)
		key = self.images[idx]
		landmarks = self.landmark_data[key]
		kpt = np.array(landmarks)

		#load and preprocess image
		image, tform  = self.preprocess_image(img_path, kpt)

		#prepare landmarks
		cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
		cropped_kpt = cropped_kpt[:,:2] 
		
		# normalized kpt
		cropped_kpt[:,0] = cropped_kpt[:,0]
		# cropped_kpt[:,:2] = cropped_kpt[:,:2]/224 * 2  - 1
		cropped_kpt[:,:2] = cropped_kpt[:,:2]/224


		#normalize landmarks
		# landmarks[..., 0] = landmarks[..., 0] / og_w
		# landmarks[..., 1] = landmarks[..., 1] / og_h
		# landmarks[..., 0] = landmarks[..., 0] / self.resolution_inp
		# landmarks[..., 1] = landmarks[..., 1] / self.resolution_inp


		return image, cropped_kpt

	def bbox2point(self, left, right, top, bottom, type='bbox'):
		''' bbox from detector and landmarks are different
		'''
		if type=='kpt68':
			old_size = (right - left + bottom - top)/2*1.1
			center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
		elif type=='bbox':
			old_size = (right - left + bottom - top)/2
			center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
		else:
			raise NotImplementedError
		return old_size, center

	def preprocess_image(self, imagepath, kpt):
		image = np.array(imread(imagepath))

		# bbox, bbox_type = self.face_detector.run(image)
		left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
		top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
		bbox = [left,top, right, bottom]
		bbox_type = 'kpt68'
		if bbox is None:
			print('no face detected', imagepath)

		left = bbox[0]; right=bbox[2]
		top = bbox[1]; bottom=bbox[3]

		old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
		size = int(old_size*self.scale)
		src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
		
		DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
		tform = estimate_transform('similarity', src_pts, DST_PTS)
		
		image = image/255.
		dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
		dst_image = dst_image.transpose(2,0,1)


		# return {'image': torch.tensor(dst_image).float(),
				# 'imagename': imagename,
				# 'tform': torch.tensor(tform.params).float(),
				# 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
				# }

		return torch.tensor(dst_image).float(), tform

				
	
def get_transforms():
	image_transforms = { 
		'train': A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
		]),

		'valid': A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
			]),


		'test': A.Compose([
			A.Resize(224, 224),
			A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
			ToTensorV2()
			])
	}

	return image_transforms

def get_dataloader(data_dir, batch_size, split, num_images=None):
	# Load the Data
	image_transforms = get_transforms()

	#select how many images to train with. To overfit on 1 image, set num_images = 1
	if num_images is not None: 
		index = [0, int(num_images)]
	else:
		index = None

	dataset = custom_dataset(root_dir=data_dir, transform=image_transforms[split], index=index)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

	return dataloader
