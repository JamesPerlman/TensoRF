
from argparse import ArgumentParser
import os
from pathlib import Path
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_ray_directions
from nerf_project import NeRFProject
from opt import config_parser

import numpy as np
import imageio
import json
from renderer import *
from train import SimpleSampler
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")
renderer = OctreeRender_trilinear_fast

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--project", required=True, help="Path to project folder")
    parser.add_argument("--checkpoint", required=True, help="Number of iterations to use for rendering")
    parser.add_argument("--json", required=True, help="Path to render.json")
    parser.add_argument("--frames", required=True, help="Path to a folder to save output frames in")

    return parser.parse_args()


C2W_TRANSFORM = torch.Tensor(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

def transform_matrix_to_c2w(transform_matrix):
    return torch.Tensor(np.array(transform_matrix)) @ C2W_TRANSFORM

@torch.no_grad()
def render(args):
    project_path = Path(args.project)
    json_path = Path(args.json)

    if not project_path.exists():
        print(f"project_path {project_path} does not exist")
        exit()
    
    if not json_path.exists():
        print(f"json_path {json_path} does not exist")
        exit()

    dataset = NeRFProject(project_path)

    ckpt = torch.load(dataset.get_checkpoint_path(args.checkpoint), map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)

    # load frames
    render_data: dict
    with open(json_path, 'r') as json_file:
        render_data = json.load(json_file)
    
    frames_path = Path(args.frames)
    evaluation_path(dataset, tensorf, render_data, OctreeRender_trilinear_fast, savePath=str(frames_path.absolute()))
    

@torch.no_grad()
def evaluation_path(
        test_dataset,
        tensorf,
        render_data,
        renderer,
        savePath=None,
        N_vis=5,
        prtx='',
        N_samples=-1,
        white_bg=False,
        ndc_ray=False,
        compute_extra_metrics=True,
        device='cuda'
    ):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    
    W = int(render_data["w"])
    H = int(render_data["h"])
    focal_x = render_data["fl_x"]
    focal_y = render_data["fl_y"]
    cx = render_data["cx"]
    cy = render_data["cy"]

    for idx, frame in tqdm(enumerate(render_data["frames"])):


        c2w = torch.FloatTensor(transform_matrix_to_c2w(frame["transform_matrix"]))
        ray_directions = get_ray_directions(H, W, [focal_x, focal_y], center=[cx, cy])
        ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

        rays_o, rays_d = get_rays(ray_directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = parse_args()
    render(args)

