from celery import shared_task
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

        result = generate_picture.delay(3, 4)
        print(result.get())
        print(result.status)
        return {"result_id": result.id}

    return render_template('home.html', user=current_user)



@views.route('/storage', methods=['GET', 'POST'])
@login_required
def storage():
    return render_template('storage.html', user=current_user)

@shared_task(ignore_result=False)
def generate_picture(x, y) -> int:

    return x+y