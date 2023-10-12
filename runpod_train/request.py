# this requires the installation of runpod-python
# with `pip install runpod-python` beforehand
import runpod

runpod.api_key = "NMM8GYVA0GGGCZL4D9WV7D4NKSRNYCLHQMDOE2UJ" # you can find this in settings

endpoint = runpod.Endpoint("3pec98zz01c2my")

run_request = endpoint.run(
    {   
        "train_batch_size": 4,
        "max_train_steps": 10,
        "lr_schedler": "linear",
        "extra": '''--optimizer_args scale_parameter=False relative_step=False warmup_init=False --network_train_unet_only''',
        "text_lr": 0.0004,
        "unet_lr": 0.0004,
        "model_id": "test_model",
        "ckpt_base": "models/sd_xl_base_1.0.safetensors",
        "optimizer": "Adafactor",
        "reg_token": "person",
        "sty_token": "<s1> person",
        "class_data": "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/k-faces-small-flat.zip",
        "output_dir": "output/locon",
        "resolution": 1024,
        "style_data": "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/1024_inpaint_merged.zip",
        "class_token": "<hoge> person",
        "network_algo": "locon",
        "prior_loss_w": 1,
        "instance_data": "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/sample1-augmented.zip",
        "network_module": "lycoris.kohya",
        "num_repeat_ins": 24,
        "num_repeat_sty": 1
    }
)

#print(run_request.id())
print(run_request.status())

print(run_request)
print(run_request.output())