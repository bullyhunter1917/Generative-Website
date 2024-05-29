from flask import Blueprint, render_template, request, current_app
from flask_login import current_user, login_required

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST':

        text_input = request.form.get('inputText')
        picture_input = request.form.get('imageInput')

        print(text_input)

        return {"result_id": text_input, "picture_input": picture_input}

    return render_template('home.html', user=current_user)

@views.route('/storage', methods=['GET', 'POST'])
@login_required
def storage():
    return render_template('storage.html', user=current_user)