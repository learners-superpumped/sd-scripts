
sdxl-1.0:
	mkdir -p models
	cd models && wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

leosam:
	mkdir -p models
	cd models && wget https://civitai.com/api/download/models/150851

setup:
	pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
	pip install --upgrade -r requirements.txt
	pip install xformers==0.0.20
	pip install bitsandbytes==0.41.0

dataset:
	mkdir -p dataset
	wget https://storage.googleapis.com/stable-diffusion-server-dev/dataset/kr.zip
	unzip kr.zip -d dataset/kr
	wget https://storage.googleapis.com/stable-diffusion-server-dev/dataset/us.zip
	unzip us.zip -d dataset/us
	wget https://storage.googleapis.com/stable-diffusion-server-dev/dataset/small-kr.zip
	unzip small-kr.zip -d dataset/small-kr
	wget https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip
	unzip b6ntuv40b6result.zip -d dataset/augmented-kr 