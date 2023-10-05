all: pose
CONTAINER_NAME = sdxl-locon-inference

leosam:
	mkdir -p models
	cd models && wget https://civitai.com/api/download/models/150851 -O leosam.safetensors

sdxl-1.0:
	mkdir -p models
	cd models && wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
	
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

magictouch:
	cd models && wget https://storage.cloud.google.com/stable-diffusion-model-repository-asia/v1/production/magic_touch.tflite

selfie_multiclass:
	cd models && wget https://storage.cloud.google.com/stable-diffusion-model-repository-asia/v1/production/selfie_multiclass_256x256.tflite