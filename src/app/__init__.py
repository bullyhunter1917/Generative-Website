from flask import Flask
import secrets
from os import path


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = secrets.token_urlsafe(16)
    # app.config['SD_MODEL'] = load_model_from_config()

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app