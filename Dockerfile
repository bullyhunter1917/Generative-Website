FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt
COPY / /app

RUN chmod +x /app/gunicorn.sh

CMD ["bash", "gunicorn.sh"]