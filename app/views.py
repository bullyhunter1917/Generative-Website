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
    style1 = request.form.get('style1')
    style2 = request.form.get('style2')
    alpha = float(request.form.get('alpha'))/1000.0
    strength = float(request.form.get('strength'))
    ddim_steps = int(request.form.get('ddim_steps'))
    n_samples = int(request.form.get('n_samples'))
    n_iter = int(request.form.get('n_iter'))


    if only_text:
        generate_picture(text, None, only_text)
    else:
        if 'files[]' not in request.files:
            print('return msg')
            return jsonify(result='', msg='Attach a file')

        file = request.files['files[]']
        if file and allowed_file(file.filename):
            if text == "" and (style1=='' or style2==''):
                return jsonify(result='', msg='If one of styles is None please add Prompt')

            filename = secure_filename(file.filename)

            print(filename)

            file.save(UPLOAD_FOLDER + '/' + filename)

            # generate_picture(text, filename, only_text)
            return jsonify(result='', msg='Generated picture')
        else:
            return jsonify(result='', msg='Accepted files: png, jpg, jpeg')


    return "done"
