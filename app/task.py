from multiprocessing import Lock, Queue
import time

from flask import jsonify

from pytorch_lightning import seed_everything
import torch
from torchvision.utils import make_grid
from einops import rearrange, repeat
from omegaconf import OmegaConf
from transformers import CLIPTokenizerFast, CLIPTextModel
from .ldm.util import instantiate_from_config
from .ldm.ddim import DDIMSampler
import numpy as np
import os
import PIL

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
OUTPATH = 'app/static/generated'

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


tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
tokenizer.add_tokens(["<charcoal-style>", "<hyperpop-style>", "<martyna-style>", "<klee-style>", "<abstract-style>", "<graffiti-style>"])
clipModel = CLIPTextModel.from_pretrained("ATTENSHONE/styletransfer")
config = OmegaConf.load('/home/saide_kamila/my_proj_env/bin/app/configs/stable-diffusion/v1-inference.yaml')
model = load_model_from_config(config, "/home/saide_kamila/my_proj_env/bin/app/models/sd/sd-v1-4.ckpt")
sampler = DDIMSampler(model)

lock = Lock()


def start(text, picture, only_text):
    lock.acquire()

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
    precision_scope = "autocast"
    content_dir = f'app/static/uploads/{picture}'
    outdir = "outputs/img2img-samples"
    seed_everything(seed)

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    content_image = load_img(content_dir).to(device)

    init_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))


    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    t_enc = int(strength * ddim_steps)

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in range(n_iter):
                    for prompts in data:
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        inputs = tokenizer([prompt], padding='max_length', return_tensors='pt')
                        c = clipModel(**inputs)['last_hidden_state'].to(device)
                        inputs2 = tokenizer([style2], padding='max_length', return_tensors='pt')
                        c2 = clipModel(**inputs2)['last_hidden_state'].to(device)

                        c = alpha * c + (1 - alpha) * c2

                        print(c.shape)

                        # img2img

                        # stochastic encode
                        # z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))

                        # stochastic inversion
                        t_enc = int(strength * 1000)
                        x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc] * batch_size).to(device))
                        model_output = model.apply_model(x_noisy, torch.tensor([t_enc] * batch_size).to(device), c)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device), noise=model_output, use_original_steps=True)

                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, )
                        print(z_enc.shape, uc.shape, t_enc)

                        # txt2img
                        #             noise  =torch.randn_like(content_latent)
                        #             samples, intermediates =sampler.sample(ddim_steps,1,(4,512,512),c,verbose=False, eta=1.,x_T = noise,
                        #    unconditional_guidance_scale=scale,
                        #    unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            # base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = PIL.Image.fromarray(grid.astype(np.uint8))
                output.save(os.path.join(OUTPATH,  prompt + picture))
                # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                # grid_count += 1

    return "done"

def generate_picture(text, picture, only_text):
    res_picture = start(text, picture, only_text)
    return res_picture
