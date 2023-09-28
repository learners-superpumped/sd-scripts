from diffusers import StableDiffusionXLPipeline
import torch
from safetensors import safe_open
from safetensors.torch import load_file
import time
import json
from lycoris.kohya import LycorisNetwork
from lycoris.config import PRESET


def create_network_from_weights(multiplier, file, vae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    unet_loras = {}
    te_loras = {}
    LycorisNetwork.apply_preset(PRESET["full"])
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if lora_name.startswith(LycorisNetwork.LORA_PREFIX_UNET):
            unet_loras[lora_name] = None
        elif lora_name.startswith(LycorisNetwork.LORA_PREFIX_TEXT_ENCODER):
            te_loras[lora_name] = None
    
    for name, modules in unet.named_modules():
        lora_name = f'{LycorisNetwork.LORA_PREFIX_UNET}_{name}'.replace('.','_')
        if lora_name in unet_loras:
            unet_loras[lora_name] = modules

    for text_encoder in text_encoders:
        for name, modules in text_encoder.named_modules():
            lora_name = f'{LycorisNetwork.LORA_PREFIX_TEXT_ENCODER}_{name}'.replace('.','_')
            if lora_name in te_loras:
                te_loras[lora_name] = modules
    network = LycorisNetwork(text_encoders, unet)
    network.unet_loras = []
    network.text_encoder_loras = []
    
    for lora_name, orig_modules in unet_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.unet_loras.append(module)
    
    for lora_name, orig_modules in te_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.text_encoder_loras.append(module)
    
    for lora in network.unet_loras + network.text_encoder_loras:
        lora.multiplier = multiplier
    
    return network, weights_sd

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_single_file(
    "./models/leosam.safetensors", torch_dtype=torch.float16, local_files_only=True,
).to("cuda")

# loha link : https://civitai.com/models/111594/sd-xl09-loha-pearly-gates-concept
# lora link : https://civitai.com/models/112904/arcane-style-lora-xl09
# lycoris link : https://civitai.com/models/108011/fcbodybuildingxl-10-for-sdxl

lora_model = "output/locon-kr-leosam-768-32/test.safetensors"
lora_strength = 1

weights_sd = safe_open(lora_model, framework="pt")
network_args = weights_sd.metadata()
print(network_args)
weights_sd = None
weights_sd = load_file(lora_model)
# print("weights_sd")
# print(weights_sd.keys())
# for key in weights_sd.keys():
#     if "text" in key:
#         print(key)
# exit()
try:
    ss_network_args_dict = json.loads(
        network_args['ss_network_args'])
    if 'algo' in ss_network_args_dict:
        algo = ss_network_args_dict['algo']

except Exception as e:
    try:
        algo = network_args['ss_network_module']
        if algo == "networks.lora":
            algo = "lora"
    except Exception as e:
        algo = "lora"
    print(e)
    print("Error: could not load ss_network_args")

if algo == "lora":
    pipe.load_lora_weights(".", weight_name=lora_model)
    #network = lora
else:
    # for SDXL two text_encoders
    network, weights_sd = create_network_from_weights(
        multiplier=lora_strength,
        file=lora_model,
        vae=pipe.vae,
        text_encoders=[pipe.text_encoder, pipe.text_encoder_2],
        unet=pipe.unet,
        weights_sd=weights_sd,
        for_inference=True,
        algo=algo
    )
    network.apply_to([pipe.text_encoder, pipe.text_encoder_2], pipe.unet, True, True)
    info = network.load_state_dict(weights_sd, False)
    network.to("cuda", dtype=torch.float16)

# create an image
generator = torch.Generator("cuda").manual_seed(0)
prompt = "leogirl, hoge girl"

image = pipe(prompt=prompt,generator=generator).images[0]
image.save(time.strftime("%Y%m%d_%H%M%S") + ".png")