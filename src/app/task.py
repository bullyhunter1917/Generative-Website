from multiprocessing import Lock, Queue
import time

import torch
from omegaconf import OmegaConf
from transformers import CLIPTokenizerFast
from .ldm.util import instantiate_from_config
from .ldm.ddim import DDIMSampler

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
    return model

model = load_model_from_config('configs/stable-diffusion/v1-inference.yaml', "./models/sd/sd-v1-4.ckpt")
lock = Lock()
queue = Queue(10)

def start(text, picture, only_text):
    lock.acquire()

    try:
        # TODO do work
        for i in range(0,10):
            print(f'did work {i} for {text}')
            time.sleep(1)
    finally:
        lock.release()
        return f'{text} is done'

def generate_picture(text, picture, only_text):
    queue.put((text, picture, only_text))
    res_picture = start(text, picture, only_text)
    queue.get()
    return res_picture

def where_in_queue():
    return queue.qsize()