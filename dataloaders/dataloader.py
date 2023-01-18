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
		
		images = os.listdir(os.path.join(root_dir + 'images'))
		images.sort()
		landmark_file = os.path.join(root_dir, 'all_data.json')
		with open(landmark_file, 'r') as f:
			self.landmark_data = json.load(f)
			
		if index is not None:
			self.images = images[index[0]:index[1]]
			# self.images = images[11:12]
	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		#get image
		img_name = os.path.join(self.root_dir + 'images/', self.images[idx])
		image = cv2.imread(img_name)
		og_h, og_w, _ = image.shape

		# transform
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.transform:
			augmented = self.transform(image=image)
			image = augmented['image']

		#get label (landmarks)
		key = str(int(self.images[idx].split('.')[0]))
		landmarks = self.landmark_data[key]['face_landmarks']
		landmarks = torch.tensor(landmarks, dtype=torch.float32)
		print('landmarks: ', landmarks.shape)
		#normalize landmarks
		landmarks[..., 0] = landmarks[..., 0] / og_w
		landmarks[..., 1] = landmarks[..., 1] / og_h
		
		return image, landmarks
	
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
