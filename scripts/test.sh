python sdxl_minimal_inference.py \
    --ckpt_path=models/leosam.safetensors \
    --prompt="leogirl, realistic Documentary photography, hoge girl" \
    --negative_prompt="(worst quality, low quality, cgi, bad eye, worst eye, illustration, cartoon), deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, open mouth" \
    --output_dir="inference_results" \
    --lora_weights="output/text-encoder-kr-leosam-768-128/test.safetensors"
