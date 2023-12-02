import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Test Restormer/MIRNet-v2 on your own images')
parser.add_argument('--model', required=True, type=str, help='Model to use', choices=['restormer', 'mirnetv2'])
parser.add_argument('--input_dir', default='./demo/degraded/', type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./demo/restored/', type=str, help='Directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Motion_Deblurring', 'lowlight_enhancement'])
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(task, parameters, model_type):
    if model_type == 'restormer':
        weights = os.path.join('Restormer','Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif model_type == 'mirnetv2':
        weights = os.path.join('MIRNetv2','Enhancement', 'pretrained_models', 'enhancement_lol.pth')
    return weights, parameters

task = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir, task)

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.' + ext)))
    files = natsorted(files)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters

if args.model == 'restormer':
    parameters = {
    'inp_channels': 3, 
    'out_channels': 3, 
    'dim': 48, 
    'num_blocks': [4, 6, 6, 8],
    'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'dual_pixel_task': False
    }
    weights, parameters = get_weights_and_parameters(task, parameters, 'restormer')
    load_arch = run_path(os.path.join('Restormer','basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)
elif args.model == 'mirnetv2':
    parameters = {
    'inp_channels':3,
    'out_channels':3, 
    'n_feat':80,
    'chan_factor':1.5,
    'n_RRG':4,
    'n_MRB':2,
    'height':3,
    'width':2,
    'bias':False,
    'scale':1,
    'task': task
    }
    weights, parameters = get_weights_and_parameters(task, parameters, 'mirnetv2')
    load_arch = run_path(os.path.join('MIRNetv2','basicsr', 'models', 'archs', 'mirnet_v2_arch.py'))
    model = load_arch['MIRNet_v2'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 8 if args.model == 'restormer' else 4

print(f"\n ==> Running {task} with weights {weights}\n ")

with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if task == 'Gaussian_Gray_Denoising' and args.model == 'restormer':
            img = load_gray_img(file_)
        else:
            img = load_img(file_)

        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            ## Testing on the original resolution image
            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0 if args.model == 'restormer' else tile % 4 == 0, "tile size should be multiple of 8" if args.model == 'restormer' else "tile size should be multiple of 4"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :height, :width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]

        if task == 'Gaussian_Gray_Denoising' and args.model == 'restormer':
            save_gray_img((os.path.join(out_dir, f + '_restormer.png')), restored)
        else:
            save_img((os.path.join(out_dir, f + '_restormer.png')), restored)

        torch.cuda.empty_cache()

        # MIRNet-v2 execution
        if args.model == 'restormer':
            # Set parameters for MIRNet-v2
            mirnet_parameters = {
                'inp_channels': 3,
                'out_channels': 3,
                'n_feat': 80,
                'chan_factor': 1.5,
                'n_RRG': 4,
                'n_MRB': 2,
                'height': 3,
                'width': 2,
                'bias': False,
                'scale': 1,
                'task': task
            }

            # Set the path for MIRNet-v2 weights
            mirnet_weights, mirnet_parameters = get_weights_and_parameters(task, mirnet_parameters, 'mirnetv2')

            # Load MIRNet-v2 architecture
            mirnet_arch_path = os.path.join('MIRNetv2', 'basicsr', 'models', 'archs', 'mirnet_v2_arch.py')
            mirnet_arch = run_path(mirnet_arch_path)
            mirnet_model = mirnet_arch['MIRNet_v2'](**mirnet_parameters)

            # Move the MIRNet-v2 model to the same device as Restormer
            mirnet_model.to(device)

            # Load MIRNet-v2 weights
            mirnet_checkpoint = torch.load(mirnet_weights)
            mirnet_model.load_state_dict(mirnet_checkpoint['params'])
            mirnet_model.eval()

            # Process the images with MIRNet-v2
            input_mirnet = restored  # Use the output of Restormer as input for MIRNet-v2

            input_mirnet = torch.from_numpy(input_mirnet).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

            # Pad the input if not_multiple_of 8
            height_mirnet, width_mirnet = input_mirnet.shape[2], input_mirnet.shape[3]
            H_mirnet, W_mirnet = ((height_mirnet + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width_mirnet + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh_mirnet = H_mirnet - height_mirnet if height_mirnet % img_multiple_of != 0 else 0
            padw_mirnet = W_mirnet - width_mirnet if width_mirnet % img_multiple_of != 0 else 0
            input_mirnet = F.pad(input_mirnet, (0, padw_mirnet, 0, padh_mirnet), 'reflect')

            restored_mirnet = mirnet_model(input_mirnet)

            restored_mirnet = torch.clamp(restored_mirnet, 0, 1)

            # Unpad the output
            restored_mirnet = restored_mirnet[:, :, :height_mirnet, :width_mirnet]

            restored_mirnet = restored_mirnet.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored_mirnet = img_as_ubyte(restored_mirnet[0])

            if task == 'Gaussian_Gray_Denoising' and args.model == 'restormer':
                save_gray_img((os.path.join(out_dir, f + '_mirnet.png')), restored_mirnet)
            else:
                save_img((os.path.join(out_dir, f + '_mirnet.png')), restored_mirnet)

    print(f"\nRestored images with Restormer and MIRNet-v2 are saved at {out_dir}")
