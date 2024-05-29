FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN pip install flask celery redis flask-sqlalchemy flask-login torch torchvision torchaudio pytorch-lightning omegaconf transformers einops pillow taming-transformers-rom1504 transformers kornia openai-clip

WORKDIR /app