# Generative-Website

To run server on your local machine
- git clone
- install libraries from requirements.txt file
- go to Generative-Website dir
- run 'wget -P app/models/sd/ https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt' command to download stable diffusion model
- run wsgi.py file to start server (if you want to run model on gpu make sure u have 11GB or more of VRAM)