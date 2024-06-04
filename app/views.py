import os.path

from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, jsonify, flash
from .task import generate_picture

UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    print("HELLO")

    return render_template('home.html')


@views.route('/_generate', methods=['POST'])
def generate():

    # if 'file' not in request.files:
    #     print('NO FILE')

    text = request.form.get('inputText')
    # on - tylko tekst, None - obraz i tekst
    only_text = request.form.get('onlyText')

    if only_text:
        generate_picture(text, None, only_text)
    else:
        if 'files[]' not in request.files:
            print('return msg')
            return jsonify(result='', msg='Proszę załączyć grafikę')

        file = request.files['files[]']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(UPLOAD_FOLDER + '/' + filename)

            generate_picture(text, filename, only_text)
        else:
            return jsonify(result='', msg='Niepoprawny plik, prosze podać png, jpg lub jpeg')


    return "done"
