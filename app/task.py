from multiprocessing import Lock, Queue
import time

from pytorch_lightning import seed_everything
import torch
from omegaconf import OmegaConf
from transformers import CLIPTokenizerFast, CLIPTextModel
from .ldm.util import instantiate_from_config
from .ldm.ddim import DDIMSampler
import numpy as np
import PIL
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_from_config(config, ckpt, verbose=False):
    print(f'torch version: {torch.__version__}')
    print(f'Is cuda available? {torch.cuda.is_available()}')
    print(f"Loading model from {ckpt} on {device}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    print('Loading model: Done')
    return model


def load_img(path):
    image = PIL.Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


# tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer.add_tokens(["<charcoal-style>", "<hyperpop-style>", "<martyna-style>", "<klee-style>", "<abstract-style>", "<graffiti-style>"])
# clipModel = CLIPTextModel.from_pretrained("ATTENSHONE/styletransfer")
# config = OmegaConf.load('/home/saide_kamila/my_proj_env/bin/app/configs/stable-diffusion/v1-inference.yaml')
# model = load_model_from_config(config, "/home/saide_kamila/my_proj_env/bin/app/models/sd/sd-v1-4.ckpt")
#sampler = DDIMSampler(model)

lock = Lock()


def start(text, picture, only_text):
    alpha = 1.0
    prompt = "<klee-style>"
    style2 = "<hyperpop-style>"

    ddim_steps = 50
    strength = 0.7
    ddim_eta = 0.0
    n_iter = 1
    C = 4
    f = 8
    n_samples = 1
    n_rows = 0
    scale = 10.0
    seed = 23
    precision = "autocast"
    # content_dir = "content/new_york.jpg"
    # content_dir = "content/woman.jpg"
    content_dir = "app/static/uploads/picture"
    outdir = "outputs/img2img-samples"
    seed_everything(seed)

    batch_size = n_samples

    content_image = load_img(content_dir).to(device)

    content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))


    lock.acquire()
    print(picture)

    try:
        if only_text == "true":
            text_out = f'Generating from text'
        else:
            text_out = f'Generating from picture'
        time.sleep(1)
    finally:
        lock.release()
        return text_out

def generate_picture(text, picture, only_text):
    res_picture = start(text, picture, only_text)
    return res_picture
