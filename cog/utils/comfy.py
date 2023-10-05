import os
import tensorflow as tf
from huggingface_hub import hf_hub_download
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import requests
from tqdm import tqdm
from comfyui.nodes import init_custom_nodes

def setup():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    print("Loading models...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "comfyui", "models")
    if not os.path.exists(f'{model_path}/vae/vae-ft-mse-840000-ema-pruned.safetensors'):
        hf_hub_download("stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.safetensors", local_dir=f"{model_path}/vae", local_dir_use_symlinks=False)

    if not os.path.exists(f'{model_path}/sams/sam_vit_b_01ec64.pth'):
        os.system(
            f'wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P {model_path}/sams'
        )
    cache = os.path.join(model_path, 'clipseg')
    if not os.path.exists(cache + "/models--CIDAS--clipseg-rd64-refined/snapshots"):
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache)
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache)

    def get_ext_dir(subpath=None, mkdir=False):
        dir = os.path.dirname(__file__)
        if subpath is not None:
            dir = os.path.join(dir, subpath)

        dir = os.path.abspath(dir)

        if mkdir and not os.path.exists(dir):
            os.makedirs(dir)
        return dir


    def get_installed_models():
        models_dir = f"{model_path}/taggers"
        return filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))

    def download_to_file(url, destination, is_ext_subpath=True, session=None):
        if is_ext_subpath:
            print(f'destination: {destination}')
            response = requests.get(url)
            size = int(response.headers.get('content-length', 0)) or None
            with tqdm(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
            ) as progressbar:
                with open(destination, mode='wb') as f:
                    perc = 0
                    for chunk in response.iter_content(2048):
                        f.write(chunk)


    def download_model(model, client_id):
        url = f"https://huggingface.co/SmilingWolf/{model}/resolve/main/"
        models_dir = f"{model_path}/taggers"
        download_to_file(
            f"{url}model.onnx", f"{models_dir}/{model}.onnx")
        download_to_file(
            f"{url}selected_tags.csv", f"{models_dir}/{model}.csv")
        
    controlnet_model_list = [
        "control_v11p_sd15_openpose.pth",
        "control_v11p_sd15_canny.pth",
        "control_v11f1e_sd15_tile.pth",
        "control_v11p_sd15_lineart.pth",
        "control_v11p_sd15_inpaint.pth",
        "control_v11p_sd15_softedge.pth",
        "control_v11f1p_sd15_depth.pth",
    ]
    for model in controlnet_model_list:
        controlnet_path = os.path.join(model_path, 'controlnet')
        if not os.path.exists(controlnet_path):
            os.makedirs(controlnet_path)
        if not os.path.exists(f'{model_path}/controlnet/{model}'):
            hf_hub_download(f"lllyasviel/ControlNet-v1-1", filename=model, local_dir=controlnet_path, local_dir_use_symlinks=False)
    if not os.path.exists(f'{model_path}/controlnet/ZoeD_M12_N.pt'):
        hf_hub_download(f"lllyasviel/Annotators", filename='ZoeD_M12_N.pt', local_dir=controlnet_path, local_dir_use_symlinks=False)
    emebedings_list = [
        ('EasyNegativeV2.safetensors', 'https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/embedding/EasyNegativeV2.safetensors'),
        ('CyberRealistic_Negative-neg.pt', 'https://huggingface.co/nolanaatama/embeddings/resolve/main/CyberRealistic_Negative-neg.pt'),
        ('FastNegativeV2.pt', 'https://huggingface.co/datasets/AddictiveFuture/sd-negative-embeddings/resolve/main/FastNegativeV2.pt'),
        ('NG_DeepNegative_V1_75T.pt', 'https://huggingface.co/Neburozakusu/civitai_deposit/resolve/main/NG_DeepNegative_V1_75T.pt'),
        ('bad-picture-chill-75v.pt', 'https://huggingface.co/nolanaatama/embeddings/resolve/main/bad-picture-chill-75v.pt'),
        ('badhandv4.pt', 'https://huggingface.co/nolanaatama/embeddings/resolve/main/badhandv4.pt'),
        ('EasyNegative.safetensors', 'https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors'),
        ('pureerosface_v1.pt', 'https://huggingface.co/samle/sd-webui-models/resolve/main/pureerosface_v1.pt'),
        ('realisticvision-negative-embedding.pt', 'https://huggingface.co/Heemyung/cyberrealistic/resolve/main/realisticvision-negative-embedding.pt'),
        ('ulzzang-6500.pt', 'https://huggingface.co/AnonPerson/ChilloutMix/resolve/main/ulzzang-6500.pt'),
        ('verybadimagenegative_v1.3.pt', 'https://huggingface.co/gemasai/verybadimagenegative_v1.3/resolve/main/verybadimagenegative_v1.3.pt'),
    ]
    for model, url in emebedings_list:
        if not os.path.exists(f'{model_path}/embeddings/{model}'):
            os.system(
            f'wget {url} -P {model_path}/embeddings'
            )
    directory_list = [
        'checkpoints',
        'taggers',
        'loras'
    ]
    all_models = ("wd-v1-4-moat-tagger-v2", 
                "wd-v1-4-convnext-tagger-v2", "wd-v1-4-convnext-tagger",
                "wd-v1-4-convnextv2-tagger-v2", "wd-v1-4-vit-tagger-v2")
    for dr in directory_list:
        if not os.path.isdir(f"{model_path}/{dr}"):
            os.makedirs(f"{model_path}/{dr}") 

    for model_name in all_models:
        installed = list(get_installed_models())
        if not any(model_name + ".onnx" in s for s in installed):
            download_model(model_name, None)
    ckpt_list = [
        "https://huggingface.co/jzli/CyberRealistic-3.2/resolve/main/cyberrealistic_v32.safetensors",
        "https://huggingface.co/BanKaiPls/AsianModel/resolve/main/Brav6.safetensors",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors",
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt",
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt"
    ]
    ckpt_path = f"{model_path}/checkpoints"

    for ckpt in ckpt_list:
        if ckpt.startswith('http'):
            ckpt_fname = ckpt.split("/")[-1]
            url = ckpt
            ckpt_base = f"{ckpt_path}/{ckpt_fname}"
            if not os.path.exists(ckpt_base):
                os.system(
                    f'wget {url} -P {ckpt_path}/'
                )