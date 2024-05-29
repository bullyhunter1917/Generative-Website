from flask import Flask

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import secrets
from os import path

db = SQLAlchemy()
DB_NAME = 'database.db'

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

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = secrets.token_urlsafe(16)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'

    db.init_app(app)

    from .views import views
    from .auth import auth

    from .models import User

    with app.app_context():
        db.create_all()

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')