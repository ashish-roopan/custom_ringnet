import trimesh
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '.')

from models.FLAME import FLAME
from configs.config import cfg as model_cfg


def batch_orth_proj(X, camera):
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
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flame = FLAME(model_cfg.model).to(device)


shape = torch.zeros(1, 100).to(device)
exp = torch.zeros(1, 50).to(device)
pose = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(device)
vertices, landmarks2d, landmarks3d = flame(shape_params=shape, expression_params=exp, pose_params=pose)

cam = torch.tensor([[10, 0, 0]]).to(device)
image_size = 1000
landmarks2d = batch_orth_proj(landmarks2d, cam)[:,:,:2]
landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]
landmarks2d = landmarks2d*image_size/2 + image_size/2

faces = flame.faces_tensor.detach().cpu().numpy()
mesh = trimesh.Trimesh(vertices[0].detach().cpu().numpy(), faces, process=False)
mesh.show()

img = np.zeros((image_size,image_size, 3), dtype=np.uint8)
for i, lmk in enumerate(landmarks2d[0, 60:]):
    cv2.circle(img, (int(lmk[0]), int(lmk[1])), 3, (255, 255, 255), -1)
    cv2.putText(img, str(i), (int(lmk[0]), int(lmk[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
