from ast import Try
import json
import numpy as np
import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes, load_obj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import pyrender
from trimesh.visual import texture, TextureVisuals
import cv2



class MeshInsights():
    def __init__(self):
        #open body segmentation mask
        try:
            with open('data/SMPL_seg_mask.json') as json_file:
                self.seg_mask = json.load(json_file)
        except:
            print('no seg mask found')

    def plot_uvs(self, texture_img, uvs, color=[0,0,0]):
        h, w = texture_img.shape[:2]
        for x,y in uvs:
            cv2.circle(texture_img, (int(x*w), h - int(y*h)), 1, color, -1)
            # cv2.circle(texture_img, (int(x*w), int(y*h)), 1, [0,0,0], -1)
        return texture_img
    
    
    def cluster_points(self,points, debug=False):
        data = points
        # data = points.cpu().numpy() # (N, 3)
        #before clustering
        if debug:
            print(f'data.shape {data.shape}')
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(data[:,0], data[:,1], data[:,2], s=50)
            ax.view_init(azim=200)
            plt.show()

        model = DBSCAN(eps=0.05, min_samples=5)
        model.fit_predict(data)
        pred = model.fit_predict(data)

        num_clusters = len(set(model.labels_))
        cluster_idx = model.labels_ 

        print("number of cluster found: {}".format(len(set(model.labels_))))

        #AFTER clustering
        if debug:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(data[:,0], data[:,1], data[:,2], c=model.labels_, s=50)
            ax.view_init(azim=200)
            plt.show()
            print('cluster for each point: ', model.labels_)

        clusters = []
        for i in range(num_clusters):
            cluster = data[cluster_idx == i]
            clusters.append(cluster)
            color = np.ones(cluster.shape[0]) * i
            if debug:
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(cluster[:,0], cluster[:,1], cluster[:,2], c=color , s=50)
                ax.view_init(azim=200)
                plt.show()

        return clusters

    def paint_vertices(self,vertices_index,texture=None, color=[0,0.2,0.9]):
        if texture is None:
            shape = int(cloth_vertices.shape[0])
            texture = np.ones((shape,3), dtype=np.uint8)*255
        
        #make color of corner vertices different
        texture[vertices_index] = color
        return texture

    def read_obj(self, obj_file):
        img_h, img_w = 512, 512 #texture.png dimensions
        uvs = []
        faces = []
        vertices = []
        uv_idxs = []
        with open(obj_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('vt'):
                    x = float(line.split()[1])
                    y = float(line.split()[2])
                    uvs.append([x,y])
                elif line.startswith('v'):
                    x = float(line.split()[1])
                    y = float(line.split()[2])
                    z = float(line.split()[3])
                    vertices.append([x,y,z])
                elif line.startswith('f'):
                    face = line.split()[1:]
                    try:
                        uv_idx = [int(i.split('/')[1]) - 1 for i in face]
                        face = [int(i.split('/')[0]) - 1 for i in face]
                        faces.append(face)
                        uv_idxs.append(uv_idx)
                    except:
                        face = [int(i) - 1 for i in face]
                        faces.append(face)

                
        return np.array(uvs), np.array(faces), np.array(vertices), np.array(uv_idxs)

    def order_uvs(self, faces, uv_idxs, uvs):
        map  = {} #v:uv
        faces = faces.reshape(-1)
        uv_idxs = uv_idxs.reshape(-1)
        for face , uv_idx in zip(faces, uv_idxs):
            map[face] = uv_idx
        
        num_verts = len(map)
        new_uvs = []
        for i in range(num_verts):
            # print(i, map[i])
            uv_idx = map[i]
            # print(f'uv_idx = {uv_idx}') 
            uv = uvs[uv_idx]
            new_uvs.append(uv)

        return np.array(new_uvs)

    def save_obj_with_texture(self, vertices, faces, uvs, obj_file, texture_file=None):
        print(f'texture_file = {texture_file}')
        if texture_file:
            # create mtl file
            mtl_file = obj_file.split('.')[0] + '.mtl'
            print(f'mtl_file = {mtl_file}')
            with open(mtl_file, 'w') as f:
                f.write('newmtl texture\n')
                f.write('map_Kd ' + texture_file + '\n')  
                # Test colors
                f.write('Ka 1.0 1.0 1.0\n')
                f.write('Kd 1.0 1.0 1.0\n')
                f.write('Ks 0.0 0.0 0.0\n')
                f.write('Ns 10.0\n')

          
        with open(obj_file, 'w') as f:
            if texture_file:
                f.write('# OBJ file\n')
                f.write('mtllib {}\n'.format(mtl_file.split('/')[-1]))
                f.write('usemtl texture\n')
            for v in vertices:
                f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
            for vt in uvs:
                f.write('vt {} {}\n'.format(vt[0], vt[1]))
            for face in faces:
                f.write('f {}/{} {}/{} {}/{}\n'.format(face[0]+1, face[0]+1, face[1]+1, face[1]+1, face[2]+1, face[2]+1))

    def get_flame_indices(self):
        with open('data/SMPL-X__FLAME_vertex_ids.npy', 'rb') as f:
            flame_indices = np.load(f)
        return flame_indices

    def image_to_texture(self, my_uvs, img):
        # img is PIL Image
        uvs = my_uvs
        material = texture.SimpleMaterial(image=img)    
        Texture = TextureVisuals(uv=uvs, image=img, material=None)
        return Texture

    def show_mesh(self,vertices, faces, texture=None):
        if texture is None:
            texture = np.ones((vertices.shape[0],3), dtype=np.uint8)*255
        mesh = trimesh.Trimesh(vertices, faces, vertex_colors=texture)
        mesh.show()
        return mesh

    def get_corner_clusters(self, vertices, faces, face_per_edge=[2, 3, 4], debug=False):
        #flatten vertices and find count of ewach vertices
        vertex_indices = torch.tensor(faces.reshape(-1)).long()
        uniq_verts , vertex_repetetion_count = torch.unique(vertex_indices, sorted=True, return_inverse=False, return_counts=True, dim=None) 
        vertex_repetetion_count = vertex_repetetion_count[vertex_indices].numpy()
        
        #find how many repetetions of vertices are there
        # repetetions, unique_repetetion_count = torch.unique(vertex_repetetion_count, sorted=True, return_inverse=False, return_counts=True, dim=None)
        
        # get vertices whose vertex_repetetion_count is 2 or 3
        mask = np.full(vertex_repetetion_count.shape[0], False, dtype=bool)
        for rep in face_per_edge:
            mask = mask | (vertex_repetetion_count == rep) 
        corner_vertices_index = vertex_indices[mask]
        
        #get corner vectors (x,y,z)
        corner_vertices = vertices[corner_vertices_index]
        
        #cluster corner vertices
        clusters = self.cluster_points(corner_vertices, debug=debug)
        return clusters
    
    def get_cluster_indeces(self, clusters, vertices):
        cluster_idxs = []
        #paint each clusters with different color
        for i, cluster in enumerate(clusters):
            #get index of cluster points in cloth_vertices
            #super set cordinates of vertices
            x = vertices[:,0]
            y = vertices[:,1]
            z = vertices[:,2]

            #subset cordinates(cluster points) of vertices
            sub_x = cluster[:,0]
            sub_y = cluster[:,1]
            sub_z = cluster[:,2]

            #check where each cluster points are in cloth_vertices
            cx = np.in1d(x, sub_x)
            cy = np.in1d(y, sub_y)
            cz = np.in1d(z, sub_z)

            #get index of cluster points in cloth_vertices
            idx_mask = cx & cy & cz
            cluster_idx = np.where(idx_mask == True)[0]
            cluster_idxs.append(cluster_idx)
        return cluster_idxs

    def save_trimesh_as_obj(self, mesh, filename):
        obj, texture = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_color=True, include_texture=True, return_texture=True, write_texture=True, resolver=None, digits=8, header='https://github.com/mikedh/trimesh')
        with open(filename, 'w') as f:
            f.write(obj)

    @staticmethod
    def image_to_texture(my_uvs, img):
        # img is PIL Image
        uvs = my_uvs
        material = texture.SimpleMaterial(image=img)    
        Texture = TextureVisuals(uv=uvs, image=img, material=material)
        return Texture

    def save_obj(self, vertices, faces, texture, filename):
        with open(filename, 'w') as fp:
            for v, vrgb in zip(vertices, texture):
                fp.write( 'v {:f} {:f} {:f} {:f} {:f} {:f}\n'.format( v[0], v[1], v[2], vrgb[0]/255, vrgb[1]/255, vrgb[2]/255))
            for f in faces: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1))

    def get_neck_vertices(self, body_vertices, body_faces, neck_vertex_indices):
        neck_vertices = body_vertices[neck_vertex_indices]
        neck_faces = []
        for i, face in enumerate(body_faces):
            #check if face is made up of all 3 points from neck vertices
            if np.in1d(face, neck_vertex_indices).all():
                #convert neck face vertex to start from 1
                f0 = np.where(np.in1d(neck_vertex_indices, face[0]))[0][0]
                f1 = np.where(np.in1d(neck_vertex_indices, face[1]))[0][0]
                f2 = np.where(np.in1d(neck_vertex_indices, face[2]))[0][0]
                neck_faces.append([f0, f1, f2])
        return neck_vertices, np.array(neck_faces)

    def get_cloth_vertices(self, vertices, faces):
        #find corner clusters
        corner_clusters = self.get_corner_clusters(vertices, faces, face_per_edge=[2, 3, 4], debug=False)
        cluster_indices = self.get_cluster_indeces(corner_clusters, vertices)
        # neck_vertices = corner_clusters[0]
        # neck_vertex_indices = cluster_indices[0]
        return corner_clusters , cluster_indices

    def get_body_neck_vertices(self, body_vertices, body_faces):
        #get neck mesh from body mesh
        neck_vertex_indices = self.seg_mask['neck']
        neck_vertices, neck_faces = self.get_neck_vertices(body_vertices, body_faces, neck_vertex_indices)
        #find corner clusters
        corner_clusters = self.get_corner_clusters(neck_vertices, neck_faces, debug=False, face_per_edge=[2, 3])
        cluster_indices = self.get_cluster_indeces(corner_clusters, neck_vertices)

        neck_vertices = corner_clusters[1] #lower neck points
        neck_vertex_indices = cluster_indices[1]
        return neck_vertices , neck_vertex_indices
    
    def get_body_corner_points(self, body_vertices, body_faces, part):
        """
        Returns corner vertices clusters of a body part
        """
        part_indices = self.seg_mask[part]
        part_vertices = body_vertices[part_indices]
        part_faces = self.get_part_faces(body_faces, part_indices)
        #find corner clusters of a body part
        corner_clusters = self.get_corner_clusters(part_vertices, part_faces, debug=False, face_per_edge=[2, 3])
        cluster_indices = self.get_cluster_indeces(corner_clusters, part_vertices)
        return corner_clusters, cluster_indices

    def get_part_faces(self, body_faces, part_vertex_indices):
        """
        Returns  ertices and faces of a single body part
        """
        part_faces = []
        for i, face in enumerate(body_faces):
            #check if face is made up of all 3 points from neck vertices
            if np.in1d(face, part_vertex_indices).all():
                #convert neck face vertex to start from 1
                f0 = np.where(np.in1d(part_vertex_indices, face[0]))[0][0]
                f1 = np.where(np.in1d(part_vertex_indices, face[1]))[0][0]
                f2 = np.where(np.in1d(part_vertex_indices, face[2]))[0][0]
                part_faces.append([f0, f1, f2])
        return np.array(part_faces)

    def add_points_to_mesh(self, scene, points, color):
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = color/255
        tfs = np.tile(np.eye(4), (len(points), 1, 1))
        tfs[:, :3, 3] = points
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)
        return scene

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Render a mesh')
    parser.add_argument('-cth' ,'--cloth_file', type=str, default='data/Tshirt.obj')
    parser.add_argument('-body' ,'--body_file', type=str, default='data/man.obj')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda:0'
    cloth_file = args.cloth_file
    body_file = args.body_file
    insight = MeshInsights()

    #############################################################################################################################
    #FIND CLOTH CORNER VERTICES
    #############################################################################################################################

    #load cloth
    cloth_mesh = trimesh.exchange.load.load(cloth_file)
    cloth_vertices = cloth_mesh.vertices
    cloth_faces = cloth_mesh.faces
    cloth_texture = np.ones((cloth_vertices.shape[0],3), dtype=np.uint8)*255

    #view cloth
    # insight.show_mesh(cloth_vertices, cloth_faces, texture)

    #find cloth corner clusters
    clusters = insight.get_corner_clusters(cloth_vertices, cloth_faces, debug=False, face_per_edge=[2, 3, 4])
    cluster_indices = insight.get_cluster_indeces(clusters, cloth_vertices)

    #paint clusters:
    cloth_texture = np.ones((cloth_vertices.shape[0],3), dtype=np.uint8)*[255,255,255]
    for i, cluster_idx in enumerate(cluster_indices):
        color = (np.random.randint(0,255,size=3)).tolist()
        cloth_texture = insight.paint_vertices(cluster_idx, cloth_texture, color=color)

    #show cloth with clusters
    cloth_mesh_og = insight.show_mesh(cloth_vertices, cloth_faces, cloth_texture)
    cloth_neck_vertices = clusters[0]

    #############################################################################################################################
    #FIND CORNER VERTICES OF NECK
    #############################################################################################################################

    #load body
    body_mesh = trimesh.exchange.load.load(body_file)
    body_vertices = body_mesh.vertices
    body_faces = body_mesh.faces
    body_texture = np.ones((body_vertices.shape[0],3), dtype=np.uint8)*255

    #open body segmentation mask
    with open('data/SMPL_seg_mask.json') as json_file:
        seg_mask = json.load(json_file)

    #paint each part with different color
    # for part, vertex_indices in seg_mask.items():
    #     # print(part)
    #     color = (np.random.randint(0,255,size=3)).tolist()
    #     body_texture[vertex_indices] = color
    # #view body
    # insight.show_mesh(body_vertices, body_faces, body_texture)


    #get neck mesh from body mesh
    neck_vertex_indices = seg_mask['neck']
    neck_vertices, neck_faces = insight.get_neck_vertices(body_vertices, body_faces, neck_vertex_indices)
    #show neck
    # insight.show_mesh(neck_vertices, neck_faces)

    #find corner clusters
    clusters = insight.get_corner_clusters(neck_vertices, neck_faces, debug=False, face_per_edge=[2, 3])
    cluster_indices = insight.get_cluster_indeces(clusters, neck_vertices)

    #paint clusters:
    neck_texture = np.ones((neck_vertices.shape[0],3), dtype=np.uint8)*[255,255,255]
    for i, cluster_idx in enumerate(cluster_indices):
        color = (np.random.randint(0,255,size=3)).tolist()
        neck_texture = insight.paint_vertices(cluster_idx, neck_texture, color=color)
        break
    #show cloth with clusters
    # insight.show_mesh(neck_vertices, neck_faces, neck_texture)
    body_neck_vertices = clusters[0]
     
    print(body_neck_vertices)
    print(body_neck_vertices.dtype)
    print(body_neck_vertices.shape)
    #############################################################################################################################
    #plot cloth centroid and body centroid
    #############################################################################################################################

    scene = pyrender.Scene()
    cloth_mesh = pyrender.Mesh.from_trimesh(cloth_mesh_og)
    scene.add(cloth_mesh)
    body_mesh = pyrender.Mesh.from_trimesh(body_mesh)
    scene.add(body_mesh)

    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(body_neck_vertices), 1, 1))
    tfs[:, :3, 3] = body_neck_vertices
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(cloth_neck_vertices), 1, 1))
    tfs[:, :3, 3] = cloth_neck_vertices
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

    body_centroid = np.sum(body_neck_vertices, axis=0)/len(body_neck_vertices)
    cloth_centroid = np.sum(cloth_neck_vertices, axis=0)/len(cloth_neck_vertices)
    

    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(body_centroid), 1, 1))
    tfs[:, :3, 3] = body_centroid
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(cloth_centroid), 1, 1))
    tfs[:, :3, 3] = cloth_centroid
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

    pyrender.Viewer(scene, use_raymond_lighting=True)


