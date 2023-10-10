all: pose
CONTAINER_NAME = sdxl-locon-inference

env:
	conda create -n cog-sdxl-infer python=3.9

setup:
	pip install -r requirements.txt

leosam:
	mkdir -p models
	cd models && wget https://civitai.com/api/download/models/150851 -O leosam.safetensors

sdxl-1.0:
	mkdir -p models
	cd models && wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

rsp:
	mkdir -p models
	cd models && wget https://civitai.com/api/download/models/154593 -O rsp.safe
	
serve:
	cog run -p 5000 python -m cog.server.http
run:
	docker run --name $(CONTAINER_NAME) -p 5001:5000 $(CONTAINER_NAME) python -m cog.server.http
remove:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)
build:
	cog build -t $(CONTAINER_NAME)
push:
	sudo cog push r8.im/learners-superpumped/user-test-arcface-loss

download-realesrgan:
	python utils/download_base_model.py --stage production --model_name RealESRGAN_x4.pth --base_model_path ./models	
download-magic_touch:
	python utils/download_base_model.py --stage production --model_name magic_touch.tflite --base_model_path ./models	
download-selfie_multiclass:
	python utils/download_base_model.py --stage production --model_name selfie_multiclass_256x256.tflite --base_model_path ./models	
