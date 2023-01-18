import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)


class Renderer:
    def __init__(self, device):
        self.device = device
        
    
    
    def create_render(self, cam=[10, 0, 0]):
        device = self.device

        # Initialize a camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
        # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
        # R, T = look_at_view_transform(10, 0, 0)
        R, T = look_at_view_transform(cam[0], cam[1], cam[2])
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)
        
        # pnt = torch.ones([1,  3], device=device)
        # print(cameras.transform_points_screen(pnt))
        # exit()
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 
        raster_settings = RasterizationSettings(
            image_size=1024, 
            blur_radius=0.0, 
            faces_per_pixel=2, 
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # create ambient light
        ambient_light = AmbientLights(device=device, ambient_color=((1.0, 1.0, 1.0),))


        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=ambient_light
            )
        )
        return renderer


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
        #points = points + translation.unsqueeze(1)
    
        # Apply perspective distortion
        projected_points = points / points[:,:,-1].unsqueeze(-1)
    
        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    
        return projected_points[:, :, :-1]



    def project_keypoints(self, mesh_points, cam_transalation, focal_length, img_shape):
        projected_keypoints_2d = self.perspective_projection(mesh_points,
                                rotation=torch.eye(3, device="cuda:0").unsqueeze(0).expand(1, -1, -1),
                                translation=cam_transalation,
                                focal_length=focal_length,
                                camera_center=torch.div(img_shape.flip(dims=[1]), 2, rounding_mode='floor'))
        return projected_keypoints_2d