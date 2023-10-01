all: pose
CONTAINER_NAME = sdxl-locon-inference

leosam:
	mkdir -p models
	cd models && wget https://civitai.com/api/download/models/150851 -O leosam.safetensors

serve:
	cog run -p 5001 python -m cog.server.http
run:
	docker run --name $(CONTAINER_NAME) -p 5001:5000 $(CONTAINER_NAME) python -m cog.server.http
remove:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)
build:
	cog build -t $(CONTAINER_NAME)
push:
	sudo cog push r8.im/oranzino/sdxl-locon-template

regular-dataset:
	wget https://storage.googleapis.com/snow_image/kor_latent/k-faces-large-flat.zip
	unzip k-faces-large-flat.zip -d dataset/k-faces-large-flat