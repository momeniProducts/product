import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import scipy.ndimage

import mcubes
import trimesh

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


# Device configuration for GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk_size):
    """Creates a batched version of a function to process data in chunks.
    """
    if chunk_size is None:
        return fn
    def batched_fn(input_tensor):
        return torch.cat([fn(input_tensor[i:i+chunk_size]) for i in range(0, input_tensor.shape[0], chunk_size)], 0)
    return batched_fn


def run_network(input_pts, view_directions, network_fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares input coordinates and applies the neural network.
    """
    flattened_inputs = torch.reshape(input_pts, [-1, input_pts.shape[-1]])
    embedded_coords = embed_fn(flattened_inputs)

    if view_directions is not None:
        expanded_dirs = view_directions[:,None].expand(input_pts.shape)
        flat_dirs = torch.reshape(expanded_dirs, [-1, expanded_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(flat_dirs)
        embedded_coords = torch.cat([embedded_coords, embedded_dirs], -1)

    flat_outputs = batchify(network_fn, netchunk)(embedded_coords)
    outputs = torch.reshape(flat_outputs, list(input_pts.shape[:-1]) + [flat_outputs.shape[-1]])
    return outputs


def batchify_rays(flattened_rays, chunk_size=1024*32, **kwargs):
    """Renders rays in smaller batches to prevent memory overflow.
    """
    results_dict = {}
    for i in range(0, flattened_rays.shape[0], chunk_size):
        batch_result = render_rays(flattened_rays[i:i+chunk_size], **kwargs)
        for key in batch_result:
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(batch_result[key])

    results_dict = {key : torch.cat(results_dict[key], 0) for key in results_dict}
    return results_dict


def render(height, width, intrinsic_matrix, chunk_size=1024*32, rays=None, cam_to_world=None, use_ndc=True,
                  near_plane=0., far_plane=1.,
                  use_view_directions=False, static_cam=None,
                  **kwargs):
    """Main rendering function for generating images from rays or camera poses.
    """
    if cam_to_world is not None:
        # Full image rendering case
        ray_origins, ray_directions = get_rays(height, width, intrinsic_matrix, cam_to_world)
    else:
        # Use provided ray batch
        ray_origins, ray_directions = rays

    if use_view_directions:
        # Use viewing directions as input
        view_dirs = ray_directions
        if static_cam is not None:
            # Special case for visualizing view direction effects
            ray_origins, ray_directions = get_rays(height, width, intrinsic_matrix, static_cam)
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        view_dirs = torch.reshape(view_dirs, [-1,3]).float()

    shape_info = ray_directions.shape
    if use_ndc:
        # For forward-facing scenes
        ray_origins, ray_directions = ndc_rays(height, width, intrinsic_matrix[0][0], 1., ray_origins, ray_directions)

    # Create ray batch tensor
    ray_origins = torch.reshape(ray_origins, [-1,3]).float()
    ray_directions = torch.reshape(ray_directions, [-1,3]).float()

    near_plane, far_plane = near_plane * torch.ones_like(ray_directions[...,:1]), far_plane * torch.ones_like(ray_directions[...,:1])
    ray_batch = torch.cat([ray_origins, ray_directions, near_plane, far_plane], -1)
    if use_view_directions:
        ray_batch = torch.cat([ray_batch, view_dirs], -1)

    # Render rays and reshape outputs
    all_results = batchify_rays(ray_batch, chunk_size, **kwargs)
    for key in all_results:
        new_shape = list(shape_info[:-1]) + list(all_results[key].shape[1:])
        all_results[key] = torch.reshape(all_results[key], new_shape)

    extraction_keys = ['rgb_map', 'disp_map', 'acc_map']
    result_list = [all_results[key] for key in extraction_keys]
    result_dict = {key : all_results[key] for key in all_results if key not in extraction_keys}
    return result_list + [result_dict]


def render_path(render_camera_poses, hwf_params, intrinsic_matrix, chunk_size, render_kwargs, ground_truth_imgs=None, save_directory=None, render_downsample_factor=0):

    img_height, img_width, focal_length = hwf_params

    if render_downsample_factor != 0:
        # Downsample for faster rendering
        img_height = img_height // render_downsample_factor
        img_width = img_width // render_downsample_factor
        focal_length = focal_length / render_downsample_factor

    rgb_results = []
    disparity_results = []

    timer_start = time.time()
    for i, cam_pose in enumerate(tqdm(render_camera_poses)):
        print(i, time.time() - timer_start)
        timer_start = time.time()
        rgb, disp, acc, _ = render(img_height, img_width, intrinsic_matrix, chunk=chunk_size, c2w=cam_pose[:3,:4], **render_kwargs)
        rgb_results.append(rgb.cpu().numpy())
        disparity_results.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if save_directory is not None:
            rgb_8bit = to8b(rgb_results[-1])
            filename = os.path.join(save_directory, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb_8bit)

    rgb_results = np.stack(rgb_results, 0)
    disparity_results = np.stack(disparity_results, 0)

    return rgb_results, disparity_results

# KEEPING YOUR extract_mesh FUNCTION EXACTLY AS IS
def extract_mesh(
    network_fn,
    network_query_fn,
    resolution=256,
    bounding_box=([-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]),
    threshold=10,
    save_path="mesh_colored.ply",
    smooth_sigma=0.25,
):
    """
    Extracts a clean, watertight mesh from a NeRF model.

    This function implements a robust pipeline to address common issues like
    holes and floating geometry (outliers).

    Args:
        network_fn: The NeRF model (coarse or fine).
        network_query_fn: The function to query the NeRF model.
        resolution (int): The resolution of the voxel grid for density sampling.
                          Higher values lead to more detail but use more memory.
        bounding_box (list of tuples): The [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
                                     bounds of the scene to be meshed.
        threshold (float): An initial density threshold for marching cubes.
                           This will be refined automatically.
        save_path (str): The path to save the final .ply mesh file.
    """
    from scipy.ndimage import zoom, gaussian_filter
    print("🔍 Sampling density field for mesh extraction…")
    
    # Use fine network if available
    if hasattr(network_fn, 'fine'):
        network_fn = network_fn.fine
        print("✅ Using fine network for extraction")
    
    # 1) Build 3D grid of points
    x = np.linspace(*bounding_box[0], resolution)
    y = np.linspace(*bounding_box[1], resolution)
    z = np.linspace(*bounding_box[2], resolution)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).astype(np.float32)
    coords = torch.from_numpy(grid.reshape(-1, 3)).to(device)  # (N,3)

    # 2) Query densities
    print("📊 Querying density field...")
    sigmas = []
    chunk = 2**18
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk):
            pts = coords[i:i+chunk]           # (B,3)
            pts_r = pts.unsqueeze(1)          # (B,1,3)
            dummy_dirs = torch.zeros(pts.shape[0], 3, device=pts.device)
            raw = network_query_fn(
                pts_r, viewdirs=dummy_dirs, network_fn=network_fn
            )                                 # (B,1,4) or (B,1,5)
            sigma = raw[..., 3].view(-1)       # (B,)
            sigmas.append(sigma.cpu().numpy())
    volume = np.concatenate(sigmas, 0).reshape(resolution, resolution, resolution)

    # 3) Denoise density field using Gaussian smoothing
    print("🧹 Applying Gaussian smoothing...")
    volume = scipy.ndimage.gaussian_filter(volume, sigma=smooth_sigma)

    # 4) Extract mesh using Marching Cubes
    print("⚙️ Running marching cubes…")
    # With adaptive thresholding: Find appropriate threshold based on density statistics
    valid_densities = volume[volume > 0.1] # Consider only non-trivial densities
    if valid_densities.size > 0:
        # A good heuristic is to set the threshold slightly above the mean density
        mean_density = np.mean(valid_densities)
        std_density = np.std(valid_densities)
        adaptive_threshold = 1.25 * mean_density + std_density
        print(f"📊 Scene stats: Mean={mean_density:.2f}, Std={std_density:.2f}")
        print(f"✅ Adaptive threshold automatically set to: {adaptive_threshold:.2f}")
    else:
        adaptive_threshold = threshold
        print(f"⚠️ Could not determine adaptive threshold, using default: {adaptive_threshold}")
        
    try:
        verts, faces = mcubes.marching_cubes(volume, adaptive_threshold)
    except ValueError as e:
        print(f"❌ Error during Marching Cubes: {e}. Try adjusting the threshold or resolution.")
        return
    print(f"🎉 Initial mesh generated with {len(verts)} vertices and {len(faces)} faces.")

    # 5) Convert mesh vertices to world coordinates
    scale = np.array([
        bounding_box[0][1] - bounding_box[0][0],
        bounding_box[1][1] - bounding_box[1][0],
        bounding_box[2][1] - bounding_box[2][0],
    ]) / resolution
    verts = verts * scale + np.array([
        bounding_box[0][0],
        bounding_box[1][0],
        bounding_box[2][0],
    ])

    # 6) Query RGB color for each vertex
    print("🎨 Sampling vertex colors…")
    vertex_colors = []
    chunk = 2**18
    vert_coords = torch.from_numpy(verts.astype(np.float32)).to(device)
    with torch.no_grad():
        for i in range(0, vert_coords.shape[0], chunk):
            pts = vert_coords[i:i+chunk]
            pts_r = pts.unsqueeze(1)
            dummy_dirs = torch.zeros(pts.shape[0], 3, device=pts.device)
            raw = network_query_fn(pts_r, viewdirs=dummy_dirs, network_fn=network_fn)
            rgb = torch.sigmoid(raw[..., :3])
            rgb = rgb.view(-1, 3)
            vertex_colors.append(rgb.cpu().numpy())
    vertex_colors = np.concatenate(vertex_colors, 0)
    vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
    vertex_colors = (vertex_colors * 255).astype(np.uint8)


    # 7) Clean mesh: Remove disconnected components
    print("🧼 Removing outlier mesh components...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
    mesh.fill_holes()

    # Optional: Add bounding box filter for additional cleaning
    bbox_min = np.array(bounding_box)[:, 0]
    bbox_max = np.array(bounding_box)[:, 1]
    in_bbox_mask = np.all(
        (mesh.vertices >= bbox_min) & 
        (mesh.vertices <= bbox_max), axis=1)
    mesh.update_vertices(in_bbox_mask)

    print("🔪 Attempting to remove the bottom surface (floor)...")
    if len(mesh.vertices) > 0:
        z_min = mesh.bounds[0, 2]
        z_max = mesh.bounds[1, 2]
        object_height = z_max - z_min
        
        # Define the floor level based on a percentage of the object's height
        floor_level = z_min + object_height * 0.1
        # Find vertices that are below the floor level
        is_floor_vertex = mesh.vertices[:, 2] < floor_level
        # A face is a "floor face" if ALL of its vertices are floor vertices
        is_floor_face = is_floor_vertex[mesh.faces].all(axis=1)
        num_floor_faces = np.sum(is_floor_face)
        if num_floor_faces > 0:
            print(f"✅ Identified and removed {num_floor_faces} floor faces.")
            mesh.update_faces(~is_floor_face)
            mesh.remove_unreferenced_vertices()
        else:
            print("👍 No floor surface detected or removed.")

    # Split mesh into connected components and find largest component (main object)
    components = mesh.split(only_watertight=False)
    # Sort by size (largest first)
    components = sorted(components, key=lambda x: x.vertices.shape[0], reverse=True)
    # Keep only components with >5% of total vertices
    min_vertices = 0.05 * verts.shape[0]
    large_components = [comp for comp in components if comp.vertices.shape[0] > min_vertices]
    if len(large_components) == 0:
        print("⚠️ No large components found! Keeping largest one.")
        mesh = components[0]
    else:
        print(f"Keeping {len(large_components)} main components")
        mesh = trimesh.util.concatenate(large_components)

    # 8) Export the final mesh with vertex colors
    print(f"💾 Saving colored mesh to {save_path}")
    mesh.export(save_path)

    
def create_nerf_model(args):
    """Initializes the NeRF MLP model and related components.
    """
    positional_embed_fn, input_channels = get_embedder(args.multires, args.i_embed)

    input_view_channels = 0
    view_embed_fn = None
    if args.use_viewdirs:
        view_embed_fn, input_view_channels = get_embedder(args.multires_views, args.i_embed)
    output_channels = 5 if args.N_importance > 0 else 4
    skip_connections = [4]
    coarse_model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_channels, output_ch=output_channels, skips=skip_connections,
                 input_ch_views=input_view_channels, use_viewdirs=args.use_viewdirs).to(device)
    trainable_params = list(coarse_model.parameters())

    fine_model = None
    if args.N_importance > 0:
        fine_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_channels, output_ch=output_channels, skips=skip_connections,
                          input_ch_views=input_view_channels, use_viewdirs=args.use_viewdirs).to(device)
        trainable_params += list(fine_model.parameters())

    network_query_function = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=positional_embed_fn,
                                                                embeddirs_fn=view_embed_fn,
                                                                netchunk=args.netchunk)

    # Initialize optimizer
    model_optimizer = torch.optim.Adam(params=trainable_params, lr=args.lrate, betas=(0.9, 0.999))

    start_iter = 0
    base_directory = args.basedir
    experiment_name = args.expname

    ##########################

    # Load model checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        checkpoint_list = [args.ft_path]
    else:
        checkpoint_list = [os.path.join(base_directory, experiment_name, f) for f in sorted(os.listdir(os.path.join(base_directory, experiment_name))) if 'tar' in f]

    print('Found checkpoints:', checkpoint_list)
    if len(checkpoint_list) > 0 and not args.no_reload:
        checkpoint_path = checkpoint_list[-1]
        print('Loading from checkpoint:', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        start_iter = checkpoint['global_step']
        model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load model weights
        coarse_model.load_state_dict(checkpoint['network_fn_state_dict'])
        if fine_model is not None:
            fine_model.load_state_dict(checkpoint['network_fine_state_dict'])

    ##########################

    training_render_kwargs = {
        'network_query_fn' : network_query_function,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : fine_model,
        'N_samples' : args.N_samples,
        'network_fn' : coarse_model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC coordinates only suitable for LLFF-style forward-facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not using NDC coordinates!')
        training_render_kwargs['ndc'] = False
        training_render_kwargs['lindisp'] = args.lindisp

    test_render_kwargs = {key : training_render_kwargs[key] for key in training_render_kwargs}
    test_render_kwargs['perturb'] = False
    test_render_kwargs['raw_noise_std'] = 0.

    return training_render_kwargs, test_render_kwargs, start_iter, trainable_params, model_optimizer


def raw_to_outputs(raw_predictions, z_values, ray_directions, raw_noise_std=0, white_background=False, pytest=False):
    """Converts raw network predictions into meaningful rendering outputs.
    """
    raw_to_alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    distances = z_values[...,1:] - z_values[...,:-1]
    distances = torch.cat([distances, torch.Tensor([1e10]).expand(distances[...,:1].shape)], -1)

    distances = distances * torch.norm(ray_directions[...,None,:], dim=-1)

    rgb_colors = torch.sigmoid(raw_predictions[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_predictions[...,3].shape) * raw_noise_std

        # Overwrite for testing consistency
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw_predictions[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw_to_alpha(raw_predictions[...,3] + noise, distances)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb_colors, -2)

    depth_map = torch.sum(weights * z_values, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_background:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                num_samples,
                return_raw=False,
                linear_disparity=False,
                perturb_amount=0.,
                num_importance_samples=0,
                fine_network=None,
                white_background=False,
                raw_noise_std=0.,
                verbose_output=False,
                pytest=False):
    """Performs volumetric rendering along rays.
    """
    num_rays = ray_batch.shape[0]
    ray_origins, ray_dirs = ray_batch[:,0:3], ray_batch[:,3:6]
    view_directions = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near_bound, far_bound = bounds[...,0], bounds[...,1]

    t_values = torch.linspace(0., 1., steps=num_samples)
    if not linear_disparity:
        z_samples = near_bound * (1.-t_values) + far_bound * (t_values)
    else:
        z_samples = 1./(1./near_bound * (1.-t_values) + 1./far_bound * (t_values))

    z_samples = z_samples.expand([num_rays, num_samples])

    if perturb_amount > 0.:
        # Calculate sample intervals
        mid_points = .5 * (z_samples[...,1:] + z_samples[...,:-1])
        upper_bounds = torch.cat([mid_points, z_samples[...,-1:]], -1)
        lower_bounds = torch.cat([z_samples[...,:1], mid_points], -1)
        # Stratified sampling within intervals
        t_random = torch.rand(z_samples.shape)

        # Fixed random numbers for testing
        if pytest:
            np.random.seed(0)
            t_random = np.random.rand(*list(z_samples.shape))
            t_random = torch.Tensor(t_random)

        z_samples = lower_bounds + (upper_bounds - lower_bounds) * t_random

    sample_points = ray_origins[...,None,:] + ray_dirs[...,None,:] * z_samples[...,:,None]

    raw_predictions = network_query_fn(sample_points, view_directions, network_fn)
    rgb_result, disp_result, acc_result, weight_values, depth_result = raw_to_outputs(raw_predictions, z_samples, ray_dirs, raw_noise_std, white_background, pytest=pytest)

    if num_importance_samples > 0:

        rgb_coarse, disp_coarse, acc_coarse = rgb_result, disp_result, acc_result

        z_midpoints = .5 * (z_samples[...,1:] + z_samples[...,:-1])
        z_importance = sample_pdf(z_midpoints, weight_values[...,1:-1], num_importance_samples, det=(perturb_amount==0.), pytest=pytest)
        z_importance = z_importance.detach()

        z_combined, _ = torch.sort(torch.cat([z_samples, z_importance], -1), -1)
        sample_points = ray_origins[...,None,:] + ray_dirs[...,None,:] * z_combined[...,:,None]

        network_to_use = network_fn if fine_network is None else fine_network
        raw_predictions = network_query_fn(sample_points, view_directions, network_to_use)

        rgb_result, disp_result, acc_result, weight_values, depth_result = raw_to_outputs(raw_predictions, z_combined, ray_dirs, raw_noise_std, white_background, pytest=pytest)

    return_dict = {'rgb_map' : rgb_result, 'disp_map' : disp_result, 'acc_map' : acc_result}
    if return_raw:
        return_dict['raw'] = raw_predictions
    if num_importance_samples > 0:
        return_dict['rgb0'] = rgb_coarse
        return_dict['disp0'] = disp_coarse
        return_dict['acc0'] = acc_coarse
        return_dict['z_std'] = torch.std(z_importance, dim=-1, unbiased=False)

    for key in return_dict:
        if (torch.isnan(return_dict[key]).any() or torch.isinf(return_dict[key]).any()) and DEBUG:
            print(f"! [Numerical Error] {key} contains nan or inf.")

    return return_dict


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    parser.add_argument("--N_iter", type=int, default=100000, help='number of training iterations')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
            
    # Load dataset
    K_matrix = None
    if args.dataset_type == 'llff':
        images, poses, bounds, render_poses, test_indices = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf_params = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded LLFF data:', images.shape, render_poses.shape, hwf_params, args.datadir)
        if not isinstance(test_indices, list):
            test_indices = [test_indices]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            test_indices = np.arange(images.shape[0])[::args.llffhold]

        val_indices = test_indices
        train_indices = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in test_indices and i not in val_indices)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near_plane = np.ndarray.min(bounds) * .9
            far_plane = np.ndarray.max(bounds) * 1.
            
        else:
            near_plane = 0.
            far_plane = 1.
        print('NEAR FAR', near_plane, far_plane)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf_params, split_indices = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded Blender data:', images.shape, render_poses.shape, hwf_params, args.datadir)
        train_indices, val_indices, test_indices = split_indices

        near_plane = 2.
        far_plane = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf_params, K_matrix, split_indices, near_plane, far_plane = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf_params}, K: {K_matrix}')
        print(f'[CHECK HERE] near: {near_plane}, far: {far_plane}.')
        train_indices, val_indices, test_indices = split_indices

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf_params, split_indices = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf_params, args.datadir)
        train_indices, val_indices, test_indices = split_indices

        hemisphere_radius = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near_plane = hemisphere_radius-1.
        far_plane = hemisphere_radius+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to correct types
    img_height, img_width, focal_length = hwf_params
    img_height, img_width = int(img_height), int(img_width)
    hwf_params = [img_height, img_width, focal_length]

    if K_matrix is None:
        K_matrix = np.array([
            [focal_length, 0, 0.5*img_width],
            [0, focal_length, 0.5*img_height],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[test_indices])

    # Create output directory and save configuration
    base_directory = args.basedir
    experiment_name = args.expname
    os.makedirs(os.path.join(base_directory, experiment_name), exist_ok=True)
    args_file = os.path.join(base_directory, experiment_name, 'args.txt')
    with open(args_file, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        config_file = os.path.join(base_directory, experiment_name, 'config.txt')
        with open(config_file, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Initialize NeRF model
    render_kwargs_train, render_kwargs_test, start_iteration, grad_vars, optimizer = create_nerf_model(args)
    global_step = start_iteration

    bounds_dict = {
        'near' : near_plane,
        'far' : far_plane,
    }
    render_kwargs_train.update(bounds_dict)
    render_kwargs_test.update(bounds_dict)

    # Move test data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Skip training if only rendering
    if args.render_only:
        print('RENDER ONLY MODE')
        with torch.no_grad():
            if args.render_test:
                # Render test poses
                images = images[test_indices]
            else:
                # Render smooth camera path
                images = None

            output_dir = os.path.join(base_directory, experiment_name, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start_iteration))
            os.makedirs(output_dir, exist_ok=True)
            print('Render poses shape:', render_poses.shape)

            # Generate and save 3D mesh
            mesh_output_path = os.path.join(output_dir, f'mesh.ply')
            extract_mesh(
                network_fn=render_kwargs_test['network_fine'] if render_kwargs_test['network_fine'] is not None else render_kwargs_test['network_fn'],
                network_query_fn=render_kwargs_train['network_query_fn'],
                save_path=mesh_output_path
            )
            return

    # Prepare ray batching
    batch_size = args.N_rand
    use_ray_batching = not args.no_batching
    if use_ray_batching:
        # Prepare random ray batches
        print('Generating rays')
        rays = np.stack([get_rays_np(img_height, img_width, K_matrix, p) for p in poses[:,:3,:4]], 0)
        print('Ray generation complete, concatenating')
        rays_rgb = np.concatenate([rays, images[:,None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])
        rays_rgb = np.stack([rays_rgb[i] for i in train_indices], 0)
        rays_rgb = np.reshape(rays_rgb, [-1,3,3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('Shuffling rays')
        np.random.shuffle(rays_rgb)

        print('Ray preparation complete')
        batch_index = 0

    # Move training data to GPU
    if use_ray_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_ray_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    total_iterations = args.N_iter + 1
    print('Training starting')
    print('TRAIN views:', train_indices)
    print('TEST views:', test_indices)
    print('VAL views:', val_indices)

    start_iteration = start_iteration + 1
    for i in trange(start_iteration, total_iterations):
        iteration_start_time = time.time()

        # Sample random ray batch
        if use_ray_batching:
            # Random sampling from all images
            batch_data = rays_rgb[batch_index:batch_index+batch_size]
            batch_data = torch.transpose(batch_data, 0, 1)
            batch_rays, target_pixels = batch_data[:2], batch_data[2]

            batch_index += batch_size
            if batch_index >= rays_rgb.shape[0]:
                print("Shuffling data after epoch completion!")
                random_indices = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[random_indices]
                batch_index = 0

        else:
            # Random sampling from single image
            img_index = np.random.choice(train_indices)
            target = images[img_index]
            target = torch.Tensor(target).to(device)
            pose = poses[img_index, :3,:4]

            if batch_size is not None:
                rays_o, rays_d = get_rays(img_height, img_width, K_matrix, torch.Tensor(pose))

                if i < args.precrop_iters:
                    crop_height = int(img_height//2 * args.precrop_frac)
                    crop_width = int(img_width//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(img_height//2 - crop_height, img_height//2 + crop_height - 1, 2*crop_height), 
                            torch.linspace(img_width//2 - crop_width, img_width//2 + crop_width - 1, 2*crop_width)
                        ), -1)
                    if i == start_iteration:
                        print(f"[Config] Center cropping of size {2*crop_height} x {2*crop_width} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, img_height-1, img_height), torch.linspace(0, img_width-1, img_width)), -1)

                coords = torch.reshape(coords, [-1,2])
                selected_indices = np.random.choice(coords.shape[0], size=[batch_size], replace=False)
                selected_coords = coords[selected_indices].long()
                rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
                rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_pixels = target[selected_coords[:, 0], selected_coords[:, 1]]

        #####  Core training loop  #####
        rgb_output, disp_output, acc_output, extra_outputs = render(img_height, img_width, K_matrix, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        image_loss = img2mse(rgb_output, target_pixels)
        transparency = extra_outputs['raw'][...,-1]
        total_loss = image_loss
        psnr_value = mse2psnr(image_loss)

        if 'rgb0' in extra_outputs:
            image_loss0 = img2mse(extra_outputs['rgb0'], target_pixels)
            total_loss = total_loss + image_loss0
            psnr0_value = mse2psnr(image_loss0)

        total_loss.backward()
        optimizer.step()

        # Learning rate scheduling
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_learning_rate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate

        iteration_time = time.time()-iteration_start_time
        #####    End training loop    #####

        # Logging and checkpointing
        if i % args.i_weights == 0:
            checkpoint_path = os.path.join(base_directory, experiment_name, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print('Saved checkpoint at:', checkpoint_path)

        if i % args.i_video == 0 and i > 0:
            # Render video and mesh
            with torch.no_grad():
                rendered_rgbs, rendered_disps = render_path(render_poses, hwf_params, K_matrix, args.chunk, render_kwargs_test)
            print('Rendering complete, saving results:', rendered_rgbs.shape, rendered_disps.shape)
            video_base = os.path.join(base_directory, experiment_name, '{}_spiral_{:06d}_'.format(experiment_name, i))
            imageio.mimwrite(video_base + 'rgb.mp4', to8b(rendered_rgbs), fps=30, quality=8)
            imageio.mimwrite(video_base + 'disp.mp4', to8b(rendered_disps / np.max(rendered_disps)), fps=30, quality=8)

            # Save 3D mesh
            mesh_path = os.path.join(base_directory, experiment_name,
                                     f'{experiment_name}_mesh_{i:06d}.ply')
            extract_mesh(
                network_fn=render_kwargs_test['network_fine'] if render_kwargs_test['network_fine'] is not None else render_kwargs_test['network_fn'],
                network_query_fn=render_kwargs_train['network_query_fn'],
                save_path=mesh_path
            )

        if i % args.i_testset == 0 and i > 0:
            test_output_dir = os.path.join(base_directory, experiment_name, 'testset_{:06d}'.format(i))
            os.makedirs(test_output_dir, exist_ok=True)
            print('Test poses shape:', poses[test_indices].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[test_indices]).to(device), hwf_params, K_matrix, args.chunk, render_kwargs_test, gt_imgs=images[test_indices], savedir=test_output_dir)
            print('Test set saved')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {total_loss.item()}  PSNR: {psnr_value.item()}")
      
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()