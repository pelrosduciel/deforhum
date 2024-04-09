#@markdown **Environment Setup**
import subprocess, time, gc, os, sys

def setup_environment():
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'
    
    if 'google.colab' in str(ipy):
        start_time = time.time()
        packages = [
            'xformers==0.0.24',
            'einops==0.4.1 pytorch-lightning==1.7.7 torchdiffeq==0.2.3 torchsde==0.2.5',
            'ftfy timm transformers open-clip-torch omegaconf torchmetrics==0.11.4',
            'safetensors kornia accelerate jsonmerge matplotlib resize-right',
            'scikit-learn numpngw pydantic'
        ]
        for package in packages:
            print(f"..installing {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + package.split())
        if not os.path.exists("deforum-stable-diffusion"):
            subprocess.check_call(['git', 'clone', '-b', '0.7.1', 'https://github.com/deforum-art/deforum-stable-diffusion.git'])
        else:
            print(f"..deforum-stable-diffusion already exists")
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend(['deforum-stable-diffusion/','deforum-stable-diffusion/src',])
        end_time = time.time()
        print(f"..environment set up in {end_time-start_time:.0f} seconds")
    else:
        sys.path.extend(['src'])
        print("..skipping setup")

setup_environment()

import torch
import random
import clip
from IPython import display
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts