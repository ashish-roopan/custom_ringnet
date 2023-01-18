
import torch
import sys
sys.path.insert(0, '../')

from models.FLAME import FLAME
from configs.config import cfg as model_cfg




class Decoder():
	def __init__(self, device):
		self.flame = FLAME(model_cfg.model).to(device)

	def decode(self, shape, pose, exp, cam, image_size=256):
		vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
		
		landmarks2d = self.batch_orth_proj(landmarks2d, cam)[:,:,:2]
		landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]

		landmarks2d = landmarks2d*image_size/2 + image_size/2
		
		return vertices, landmarks2d/image_size
		# return vertices, landmarks2d

	def batch_orth_proj(self, X, camera):
		''' orthgraphic projection
			X:  3d vertices, [bz, n_point, 3]
			camera: scale and translation, [bz, 3], [scale, tx, ty]
		'''
		camera = camera.clone().view(-1, 1, 3)
		X_trans = X[:, :, :2] + camera[:, :, 1:]
		# X_trans = X[:, :, :2]

		X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
		shape = X_trans.shape
		Xn = (camera[:, :, 0:1] * X_trans)
		return Xn





