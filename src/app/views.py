from flask import Blueprint, render_template, request, jsonify, flash
from .task import generate_picture, where_in_queue, queue

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():


    return render_template('home.html')

@views.route('/_generate')
def generate():
    print("INSIDE GENERATE")
    print(queue.full())
    if queue.full():
        flash("Too much trafic on website plese try again later", category="error")
    else:
        text = request.args.get('text')
        picture = request.args.get('picture')
        only_text = request.args.get('only_text')

        return jsonify(result=generate_picture(text, picture, only_text))

@views.route('/_queue')
def position_in_queue():
    if not queue.full():
        return jsonify(result=where_in_queue())
    else:
        return jsonify(result="")