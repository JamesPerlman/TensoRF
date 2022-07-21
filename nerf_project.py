import torch,cv2
import json
import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
from dataLoader.ray_utils import get_ray_directions, get_rays

BLENDER_TO_OPENCV_MATRIX = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
])

def read_json(path: Path):
    with open(path) as f:
        return json.load(f)

class NeRFProject(Dataset):
    def __init__(self, project_path: Path):
        transforms_json_path = project_path / "transforms.json"

        self.transform = T.ToTensor()
        
        self.white_bg = True
        self.near_far = [0.1, 100.0]
        
        json = read_json(transforms_json_path)

        w = int(json['w'])
        h = int(json['h'])
        self.img_wh = [w, h]
        self.focal_x = 0.5 * w / np.tan(0.5 * json['camera_angle_x'])  # original focal length
        self.focal_y = 0.5 * h / np.tan(0.5 * json['camera_angle_y'])  # original focal length
        self.cx = json['cx']
        self.cy = json['cy']
        
        s = float(json["aabb_scale"])
        self.scene_bbox = torch.tensor([[-s, -s, -s], [s, s, s]])
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y], center=[self.cx, self.cy])
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([
            [self.focal_x, 0, self.cx],
            [0, self.focal_y, self.cy],
            [0, 0, 1]
        ]).float()

        self.image_paths = []
        self.all_depth = []

        poses = []
        all_rays = []
        all_rgbs = []

        idxs = list(range(len(json['frames'])))
        for i in tqdm(idxs, desc=f'Loading data ({len(idxs)})'):

            frame = json['frames'][i]
            pose = np.array(frame['transform_matrix']) @ BLENDER_TO_OPENCV_MATRIX
            c2w = torch.FloatTensor(pose)
            poses.append(c2w)

            image_path = project_path / frame['file_path']
            self.image_paths.append(image_path)

            img = Image.open(image_path)
            img = self.transform(img)  # (4, h, w)
            img = img.view(-1, w * h).permute(1, 0)  # (h * w, 4) RGBA
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            all_rgbs.append(img)

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            all_rays.append(torch.cat([rays_o, rays_d], 1))  # (h*w, 6)


        self.poses = torch.stack(poses)
        self.all_rays = torch.cat(all_rays, 0)
        self.all_rgbs = torch.cat(all_rgbs, 0)
        
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]
        
    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        return {
            'rays': self.all_rays[idx],
            'rgbs': self.all_rgbs[idx]
        }
