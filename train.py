from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from nerf_project import NeRFProject
from renderer import OctreeRender_trilinear_fast

from utils import N_to_reso, TVLoss, cal_n_samples

PROGRESS_REFRESH_RATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast

def parse_args():
    parser = ArgumentParser()

    # basic
    parser.add_argument("--project", type=str, required=True, help="Path to project.")
    parser.add_argument("--max_iters", type=int, default=10000, help="Number of iterations to train.")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint to load and start training from.")
    parser.add_argument("--output", type=str, default=None, help="Output file path (will be set automatically if None).")
    parser.add_argument("--save_every", type=int, default=10000, help="Save a checkpoint for every n iterations.")

    # advanced
    parser.add_argument('--model_name', type=str, default='TensorVMSplit', choices=['TensorVMSplit', 'TensorCP'])
    parser.add_argument("--batch_size", type=int, default=4096)

    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to max_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    
    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, default=[16,16,16], action="append")
    parser.add_argument("--n_lamb_sh", type=int, default=[48,48,48], action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thresh", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thresh", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shading_mode", type=str, default="MLP_Fea",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=2,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=2,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0) # ndc = normalized device coordinates
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)

    ## blender flags
    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)

    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)

    parser.add_argument("--upsamp_list",
                        type=int,
                        default=[2000,3000,4000,5500,7000],
                        action="append")

    parser.add_argument("--update_AlphaMask_list",
                        type=int,
                        default=[2000,4000],
                        action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
        
    return parser.parse_args()


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

def get_checkpoint_path(project_path: Path, n_iters: int) -> Path:
    checkpoints_path = project_path / "checkpoints"
    return checkpoints_path / f"ckpt-{n_iters}.terf.th"

def reconstruction(args):
    project_path = Path(args.project)

    dataset = NeRFProject(project_path)

    get_checkpoint_path(project_path, 0).parent.mkdir(exist_ok=True)
    
    white_bg = dataset.white_bg
    near_far = dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    print(reso_cur)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    if args.checkpoint is not None:
        print(f"LOADING CHECKPOINT {args.checkpoint}")
        checkpoint_path: Path
        try:
            checkpoint_path = get_checkpoint_path(project_path, int(args.checkpoint))

        except:
            pass
            
        if (checkpoint_path == None) or (not checkpoint_path.exists()):
            checkpoint_path = Path(args.checkpoint)
        
        ckpt = torch.load(checkpoint_path, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb,
            reso_cur,
            device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shading_mode=args.shading_mode,
            alphaMask_thres=args.alpha_mask_thresh,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=args.fea_pe,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct
        )


    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.max_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.max_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic spacei
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1
                )
            )
        ).long()
    ).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs = []

    allrays, allrgbs = dataset.all_rays, dataset.all_rgbs

    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(tensorf.n_iters, args.max_iters), miniters=PROGRESS_REFRESH_RATE, file=sys.stdout)
    for iteration in pbar:

        tensorf.n_iters = iteration

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train,
            tensorf,
            chunk=args.batch_size,
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            is_train=True
        )

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            # summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1
            # summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            # summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            # summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        # summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        # summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % PROGRESS_REFRESH_RATE == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2] < (256 ** 3): # update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.max_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.max_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        
        if iteration > 0 and iteration % args.save_every == 0:
            tensorf.save(get_checkpoint_path(project_path, iteration))
        
    final_checkpoint_path = get_checkpoint_path(project_path, args.max_iters)
    if not final_checkpoint_path.exists():
        tensorf.save(final_checkpoint_path)


# main
if __name__ == "__main__":

    # initialize
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = parse_args()

    reconstruction(args)
    