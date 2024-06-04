# Generative-Website

To run server on your local machine
- Install docker
- Pull docker image for pytorch for you graphics version
- Install docker compose plugin
- go to Generative-Website dir
- run 'wget -P WebApp/models/sd/ https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt' command to download stable diffusion model
- run 'docker compose up --build' command to start server