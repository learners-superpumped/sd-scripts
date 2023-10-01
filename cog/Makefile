all: pose

leosam:
	mkdir -p models
	cd models && wget https://civitai.com/api/download/models/150851 -O leosam.safetensors

serve:
	cog run -p 5000 python -m cog.server.http
push:
	sudo cog push r8.im/oranzino/sdxl-locon-template

regular-dataset:
	wget https://storage.googleapis.com/snow_image/kor_latent/k-faces-large-flat.zip
	unzip k-faces-large-flat.zip -d dataset/k-faces-large-flat