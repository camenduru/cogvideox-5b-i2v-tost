FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    tqdm==4.66.5 numpy==1.26.3 imageio==2.35.1 imageio-ffmpeg==0.5.1 xformers==0.0.27.post2 diffusers==0.30.3 moviepy==1.0.3 transformers==4.44.2 accelerate==0.33.0 sentencepiece==0.2.0 pillow==9.5.0 runpod && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/scheduler/scheduler_config.json -d /content/model/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/text_encoder/config.json -d /content/model/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/resolve/main/text_encoder/model-00001-of-00002.safetensors -d /content/model/text_encoder -o model-00001-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/resolve/main/text_encoder/model-00002-of-00002.safetensors -d /content/model/text_encoder -o model-00002-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/text_encoder/model.safetensors.index.json -d /content/model/text_encoder -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/tokenizer/added_tokens.json -d /content/model/tokenizer -o added_tokens.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/tokenizer/special_tokens_map.json -d /content/model/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/resolve/main/tokenizer/spiece.model -d /content/model/tokenizer -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/tokenizer/tokenizer_config.json -d /content/model/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/transformer/config.json -d /content/model/transformer -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/resolve/main/transformer/diffusion_pytorch_model-00001-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00001-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/resolve/main/transformer/diffusion_pytorch_model-00002-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00002-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/resolve/main/transformer/diffusion_pytorch_model-00003-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00003-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/transformer/diffusion_pytorch_model.safetensors.index.json -d /content/model/transformer -o diffusion_pytorch_model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/vae/config.json -d /content/model/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/model/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/configuration.json -d /content/model -o configuration.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vdo/CogVideoX-5b-I2V/raw/main/model_index.json -d /content/model -o model_index.json

COPY ./worker_runpod.py /content/worker_runpod.py
WORKDIR /content
CMD python worker_runpod.py