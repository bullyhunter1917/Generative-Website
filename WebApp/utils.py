from celery import Celery

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

def loadTokenizer():
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14", vocab_file='tokenizer/vocab.json',
                                                  merges_file='tokenizer/merges.txt')
    tokenizer.add_tokens(['<charcoal-style>', '<hyperpop-style>', '<martyna-style>'])
    return

def make_celery(app):
    celery = Celery(app.import_name)
    celery.conf.update(app.config["CELERY_CONFIG"])

    config = OmegaConf.load("WebApp/configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "WebApp/models/sd/sd-v1-4.ckpt")
    sampler = DDIMSampler(model)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery