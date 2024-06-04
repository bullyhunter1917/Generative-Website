from flask import Flask
import secrets
from os import path

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = secrets.token_urlsafe(16)

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    print("Starting server: DONE")

    return app