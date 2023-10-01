python sdxl_minimal_inference.py \
    --ckpt_path=models/leosam.safetensors \
    --prompt="leogirl, realistic Documentary photography, hoge person" \
    --negative_prompt="(worst quality, low quality, cgi, bad eye, worst eye, illustration, cartoon), deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, open mouth" \
    --output_dir="inference_results" \
    --lora_weights="output/locon-augmented-kr-leosam-512-32-3e-4-reg-2000-2-1/test.safetensors"
