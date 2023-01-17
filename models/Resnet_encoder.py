import math
import torch
import torch.nn as nn
from torchvision import models



class Linear(nn.Module):
	def __init__(self, in_features):
		super(Linear, self).__init__()
		self.linear = nn.Linear(in_features, 2048)

	def forward(self, x):
		x = self.linear(x)
		return x

class Encoder(torch.nn.Module):
	def __init__(self, device='cuda'):
		super(Encoder, self).__init__()
		
		#variables
		fc1_feat_num = 1024
		fc2_feat_num = 1024
		img_feat_num = 2048
		num_pose_param = 6
		num_shape_param = 100
		num_exp_param = 50
		num_cam_param = 3
		final_feat_num = fc2_feat_num
		reg_in_feat_num = img_feat_num + num_pose_param + num_shape_param + num_exp_param + num_cam_param 

		#encoder
		self.encoder = models.resnet50(pretrained=True)
		self.encoder.fc = Linear(in_features=self.encoder.fc.in_features)

		#initial params
		self.init_pose = torch.zeros((1, num_pose_param), device=device)
		self.init_shape = torch.zeros((1, num_shape_param), device=device)
		self.init_exp = torch.zeros((1, num_exp_param), device=device)
		# self.init_cam = torch.zeros((1, num_cam_param), device=device)
		self.init_cam = torch.tensor([[ 5.5489,  0.0186, -0.0156]], device=device)

		#decoder head for pose, shape, expression, and camera
		self.fc3 = nn.Linear(final_feat_num, num_pose_param)
		self.fc4 = nn.Linear(final_feat_num, num_shape_param)
		self.fc5 = nn.Linear(final_feat_num, num_cam_param)
		self.fc6 = nn.Linear(final_feat_num, num_exp_param)
		

		#regression layers
		self.r = nn.ReLU()
		self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
		self.drop1 = nn.Dropout()
		self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
		self.drop2 = nn.Dropout()

		# Random shit 1
		nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)
		nn.init.xavier_uniform_(self.fc4.weight, gain=0.01)
		nn.init.xavier_uniform_(self.fc5.weight, gain=0.01)
		nn.init.xavier_uniform_(self.fc6.weight, gain=0.01)
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		# 		m.weight.data.normal_(0, math.sqrt(2. / n))
		# 	elif isinstance(m, nn.BatchNorm2d):
		# 		m.weight.data.fill_(1)
		# 		m.bias.data.zero_()

	def decode_pose(self, x):
		x = self.fc3(x)
		# x = self.r(x+30) - self.r(x-30) - 30 #range [-30, 30]
		return x

	def decode_shape(self, x):
		x = self.fc4(x)
		# x = self.r(x+3) - self.r(x-3) - 3    #range [-3, 3]
		return x
	
	def decode_cam(self, x):
		x = self.fc5(x)
		# x = self.r(x+3) - self.r(x-3) - 3    #range [-3, 3]
		return x

	def decode_exp(self, x):
		x = self.fc6(x)
		# x = self.r(x+3) - self.r(x-3) - 3	#range [-3, 3]
		return x
		
	def forward(self, x, n_iter=4):
			pred_pose = self.init_pose
			pred_shape = self.init_shape
			pred_exp = self.init_exp
			pred_cam = self.init_cam

			xf = self.encoder(x)
			for i in range(n_iter):
				xc = torch.cat([xf, pred_pose, pred_shape, pred_exp, pred_cam], 1)

				xc = self.fc1(xc)
				# xc = self.drop1(xc)
				xc = self.fc2(xc)
				# xc = self.drop2(xc)

				pred_pose = self.decode_pose(xc) + pred_pose
				pred_shape = self.decode_shape(xc) + pred_shape
				pred_exp = self.decode_exp(xc) + pred_exp
				pred_cam = self.decode_cam(xc) + pred_cam

			# pred_cam = self.r(pred_cam) - self.r(pred_cam-10)
			# pred_pose = torch.clamp(pred_pose, -0.001, 0.)
			# pred_shape = torch.clamp(pred_shape, -0.001, 0.)
			# pred_exp = torch.clamp(pred_exp, -0.001, 0.)
			# pred_cam = torch.clamp(pred_cam, 8, 10)
			# pred_cam_delta = torch.clamp(pred_cam, 0., 3.)
			# pred_cam = torch.tensor([[3.0, 0.1, -0.05]], device=pred_cam.device)

			return pred_cam, pred_pose, pred_shape, pred_exp