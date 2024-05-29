from celery import shared_task
from flask import Blueprint, render_template, request, current_app
from flask_login import current_user, login_required
from .tasks import generate_from_text

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST':

        text_input = request.form.get('inputText')
        picture_input = request.form.get('imageInput')

        print(text_input)

        result = generate_from_text.delay(text_input)
        print(result.get())
        print(result.status)
        return {"result_id": result.id}

    return render_template('home.html', user=current_user)

@views.route('/cancel/<task_id>')
def cancel(task_id):
    task = generate_from_text.AsyncResult(task_id)
    task.abort()
    return "CANCELED!"

@views.route('/storage', methods=['GET', 'POST'])
@login_required
def storage():
    return render_template('storage.html', user=current_user)