"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, encode
import os
import debug

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def reconstruct_video(args):
    path_to_models = args.path_to_models
    input_videos = args.input_videos
    model_name = args.model_name
    output_folder = args.output_folder
    processed_subfolder = args.processed_subfolder

    ## 0.0) Load the model
    mode = args.mode
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    ## 0.1) Iterate over all videos in a directory
    for filename in os.listdir(input_videos):
        f_path = os.path.join(input_videos, filename)
        if not f_path.endswith(".mp4"):
            continue

        print(f"Processing {filename}")
   
        ## 1) Process the video - extract the frames from video and detected faces
        dm = TestFaceVideoDM(f_path, Path(output_folder) / filename[:-4], processed_subfolder=processed_subfolder, 
            batch_size=128, num_workers=12, face_detect_batch_size=8)
        dm.prepare_data()
        dm.setup()
        processed_subfolder = Path(dm.output_dir).name


        ## 2) Get the data loadeer with the detected faces
        dl = dm.test_dataloader()

        ## 3) Run the model on the data
        for j, batch in enumerate (auto.tqdm( dl)):
            current_bs = batch["image"].shape[0]
            img = batch
            vals = encode(emoca, img)

            for i in range(current_bs):
                name =  batch["image_name"][i]
                sample_output_folder = Path(output_folder) / filename[:-4] / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)
                save_codes(sample_output_folder, name, vals, i)
            
    print("Done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_videos', type=str, default=str(Path(gdl.__file__).parents[1] / "/assets/data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4"), 
        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail", choices=["detail", "coarse"], help="Which model to use for the reconstruction.")
    parser.add_argument('--save_images', type=str2bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail', 
        choices=["geometry_detail", "geometry_coarse", "out_im_detail", "out_im_coarse"], 
        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None, 
        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    parser.add_argument('--cat_dim', type=int, default=0, 
        help="The result video will be concatenated vertically if 0 and horizontally if 1")
    parser.add_argument('--include_rec', type=str2bool, default=True, 
        help="The reconstruction (non-transparent) will be in the video if True")
    parser.add_argument('--include_transparent', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--include_original', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--black_background', type=str2bool, default=False, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--use_mask', type=str2bool, default=True, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--logger', type=str, default="", choices=["", "wandb"], help="Specify how to log the results if at all.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reconstruct_video(args)
    print("Done")


if __name__ == '__main__':
    main()
