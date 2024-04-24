from flask import Blueprint, render_template
from flask_login import current_user, login_required

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html', user=current_user)

@views.route('/storage', methods=['GET', 'POST'])
@login_required
def storage():
    return render_template('storage.html', user=current_user)