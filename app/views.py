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
    alpha = float(request.form.get('alpha')) / 1000.0
    if only_text:
        strength = 0.7
    else:
        strength = float(request.form.get('strength'))

    ddim_steps = int(request.form.get('ddim_steps'))
    n_samples = int(request.form.get('n_samples'))
    n_iter = int(request.form.get('n_iter'))

    if only_text:
        if text == "":
            return jsonify(msg='Please provide input in only text mode', t='error')

        res = generate_picture(text, None, only_text, style1, style2, alpha, strength, ddim_steps, n_samples, n_iter)

        if n_samples == n_iter:
            return jsonify(result=res, msg='Generated picture', t='success')
        elif n_samples > n_iter:
            return jsonify(result=res, msg='Generated picture', t='success', col='')
        elif n_iter > n_samples:
            return jsonify(result=res, msg='Generated picture', t='success', row='')

    else:
        if 'files[]' not in request.files:
            return jsonify(msg='Attach a file', t='error')

        file = request.files['files[]']
        if file and allowed_file(file.filename):
            if text == "" and (style1 == '' or style2 == ''):
                return jsonify(msg='If one of styles is None please add Prompt', t='error')

            filename = secure_filename(file.filename)

            file.save(UPLOAD_FOLDER + '/' + filename)

            res = generate_picture(text, filename, only_text, style1, style2, alpha, strength, ddim_steps, n_samples,
                                   n_iter)

            if n_samples == n_iter:
                return jsonify(result=res, msg='Generated picture', t='success')

            elif n_samples > n_iter:
                return jsonify(result=res, msg='Generated picture', t='success', col='')

            elif n_iter > n_samples:
                return jsonify(result=res, msg='Generated picture', t='success', row='')

        else:
            return jsonify(msg='Accepted files: png, jpg, jpeg', t='error')

    return "done"