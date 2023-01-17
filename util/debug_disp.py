import cv2
import numpy as np
import torch
from pytorch3d.structures import Meshes

import sys
sys.path.insert(0, '../')
from models.decoder import Decoder


class Debug_diplay():
    def __init__(self, device, flame, helper, renderer):
        self.flame = flame
        self.helper = helper
        self.renderer = renderer
        self.decoder = Decoder(device)
        self.device = device
        self.mean = np.array([0.406, 0.456, 0.485])
        self.sd = np.array([0.225, 0.224, 0.229])
        self.faces = flame.faces_tensor

        #get mesh texture
        obj_filename = '/home/ashish/code/3D/DECA/outputs/anand/mica_head.obj'
        albedo_filename = '/home/ashish/code/3D/DECA/outputs/anand/anand.png'
        uvs, _, _ = helper.read_obj(obj_filename)
        self.albedo_texture = helper.create_albedo_mask_texture(albedo_filename, uvs, self.faces)
    

    def get_og_image(self, inputs, height, width):
        in_img = inputs[:1,:].detach().cpu().squeeze().permute(1,2,0).numpy()[:,:,::-1]
        in_img = ((in_img*self.sd + self.mean)*255).astype(np.uint8)
        in_img = cv2.resize(in_img, (width, height))
        return in_img

    def get_mesh_image(self, vertices, height, width):
        mesh = Meshes(verts=[vertices[0]], faces=[self.faces], textures=self.albedo_texture)
        mesh_image = self.renderer(mesh)[0, ..., :3].cpu().detach().numpy()
        mesh_image = cv2.cvtColor(mesh_image, cv2.COLOR_BGR2RGB)
        mesh_image = cv2.resize(mesh_image, (height, width))
        return mesh_image

    def debug_disp(self, model, dataloader):
        inputs, gt_landmarks = next(iter(dataloader))
        inputs = inputs.to(self.device)
        gt_landmarks = gt_landmarks.to(self.device)
        model.to(self.device)

        #run model    
        model.eval()
        with torch.no_grad():
            pred_cam, pred_pose, pred_shape, pred_exp = model(inputs)
        
        #decode model output
        vertices, projected_landmarks = self.decoder.decode(pred_shape, pred_pose, pred_exp, pred_cam)
        
        #get original image
        input_img = self.get_og_image(inputs, 512, 512)

        #get mesh image
        mesh_image = self.get_mesh_image(vertices, 512, 512)
        
        # project landmarks with custom camera
        camera = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).to(self.device)
        vertices, custom_projected_landmarks = self.decoder.decode(pred_shape, pred_pose, pred_exp, camera)
        #draw gt_landmarks on input image with green color for gt and red for predicted with increasing saturation and value
        landmark_img = np.zeros((512, 512, 3), dtype=np.uint8)       
        r = 3 
        for i in range(68):
            gt_y = int(gt_landmarks[0, i, 1]*512)
            gt_x = int(gt_landmarks[0, i, 0]*512)
            pred_y = int(projected_landmarks[0, i, 1]*512)
            pred_x = int(projected_landmarks[0, i, 0]*512)
            custom_pred_y = int(custom_projected_landmarks[0, i, 1]*512)
            custom_pred_x = int(custom_projected_landmarks[0, i, 0]*512)

            # color = np.random.randint(0, 255, size=(3,)).tolist()
            cv2.circle(input_img, (gt_x, gt_y), r, (0,255,0), -1)
            cv2.circle(landmark_img, (gt_x, gt_y), r, (0,255,0), -1)
            cv2.circle(landmark_img, (pred_x, pred_y), r, (0,0,255), -1)
            cv2.circle(mesh_image, (custom_pred_x, custom_pred_y), r, (255,0,0), -1)

            # cv2.imshow('input_img', input_img)
            # cv2.imshow('landmark_img', landmark_img)
            # cv2.waitKey(0)

        #combine images
        img = np.hstack((input_img/255, mesh_image, landmark_img))
        
        return img

