import cv2
import numpy as np
import torch
import pickle
from PIL import Image
import sys
sys.path.insert(0, '../utils') 
sys.path.insert(1,'./utils')
print('sys.path: ', sys.path)

from analyze_mesh import MeshInsights
from render import Renderer

from pytorch3d.renderer import TexturesUV
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex



class Helper:
    def __init__(self, device):
        self.device = device
        self.insight = MeshInsights()
        self.flame_masks = self.get_Flame_masks()
        Render = Renderer(device)
        self.renderer = Render.create_render()

    def get_Flame_masks(self):
        with open('data/FLAME_masks.pkl', 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        return masks

    def create_mesh_mask_texture(self, verts):
        #find the mask of the eyes of the mesh
        textures = torch.ones_like(verts)[None] * 0 # (1, V, 3)
        left_eye_mask = self.flame_masks['left_eyeball']
        right_eye_mask = self.flame_masks['right_eyeball']

        #set the color of the eyes to white
        textures[0, left_eye_mask] = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        textures[0, right_eye_mask] = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        # set the color of face to black
        face_mask = self.flame_masks['face']
        textures[0, face_mask] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        return textures

    def create_albedo_mask_texture(self, texture_filename, uvs, faces):
        with Image.open(texture_filename) as image:
            np_image = np.asarray(image.convert("RGB")).astype(np.float32)
        tex = torch.from_numpy(np_image / 255.)[None].to(self.device)
        texture = TexturesUV(maps=tex, faces_uvs=faces[None], verts_uvs=uvs)
        return texture

    def read_obj(self, obj_filename):
        uvs, faces, verts, uv_idxs = self.insight.read_obj(obj_filename)
        uvs = self.insight.order_uvs(faces, uv_idxs, uvs)
        uvs = torch.from_numpy(uvs).to(self.device).unsqueeze(0).float()
        initial_mesh = load_objs_as_meshes([obj_filename], device=self.device)
        verts = initial_mesh.verts_packed()
        faces = initial_mesh.faces_packed()
        return uvs, faces, verts

    def create_textures(self, verts, uvs, faces, mask_albedo_filename, actual_texture_filename):
        #create mesh mask 
        mesh_mask_textures = self.create_mesh_mask_texture(verts)
        #create albedo mask
        albedo_mask_texture = self.create_albedo_mask_texture(mask_albedo_filename, uvs, faces)
        #get reference mesh with actual texture
        reference_mesh_texture = self.create_albedo_mask_texture(actual_texture_filename, uvs, faces)
        return mesh_mask_textures, albedo_mask_texture, reference_mesh_texture

    def transform_vertices(self, verts):
        #scale and translate the mesh
        # new_verts = verts - torch.mean(verts, dim=0)
        new_verts = verts 
        new_verts = new_verts * 10
        new_verts = new_verts + torch.tensor([-0.0, 0.0, -1.0], device=self.device)
        return new_verts

    def draw_contour(self, img, color, bg):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 255
        ret, thresh = cv2.threshold(gray, 50, 255, 0)
        
        thresh = thresh.astype(np.uint8)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bg = cv2.drawContours(bg, contours, -1, color, 3)
        return bg

    def get_mesh_mask(self, vertices, faces, mask_textures):
        mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=mask_textures))
        mesh_mask = self.renderer(mesh)[0, ..., :3]
        alpha = self.renderer(mesh)[0, ..., 3]
        alpha = torch.stack([alpha, alpha, alpha], dim=2)
        alpha = alpha.bool()
        mesh_mask = mesh_mask * ~~alpha
        return mesh_mask
        
    def dice_loss(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

    def perspective_projection(self, points, rotation, translation,
                           focal_length, camera_center):
        """
        This function computes the perspective projection of a set of points.
        Input:
            points (bs, N, 3): 3D points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size = points.shape[0]
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:,0,0] = focal_length
        K[:,1,1] = focal_length
        K[:,2,2] = 1.
        K[:,:-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij,bkj->bki', rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:,:,-1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

        return projected_points[:, :, :-1]

        