from flask import Flask, render_template
import os

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app
