from flask import Flask

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import secrets
from os import path
from .utils import make_celery

db = SQLAlchemy()
DB_NAME = 'database.db'

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = secrets.token_urlsafe(16)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config["CELERY_CONFIG"] = {"broker_url": "redis://redis", "result_backend": "redis://redis"}

    db.init_app(app)

    celery = make_celery(app)
    celery.set_default()

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

    return app, celery

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')