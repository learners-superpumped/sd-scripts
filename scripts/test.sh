python sdxl_minimal_inference.py \
    --ckpt_path=models/sd_xl_base_1.0.safetensors \
    --prompt="a photo of hoge person" \
    --negative_prompt="((distorted eye)), NSFW, cartoon, 3d, tooth, teeth, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), asymmetric eyes, (close up), strange colours, blurry, boring, sketch, lackluster, face portrait, self-portrait, signature, letters, watermark, grayscale, low quality, worst quality, distorted hands, mutated hands, mutated finger, crooked nose, uneven teeth, face tattoo, gloomy, depressed, dots, dirty, asymmetric lips, grains, fused hands, fused heads, fused characters. foreground characters, airbrushed, glossy, photoshop" \
    --output_dir="inference_results" \
    --lora_weights="output/text-encoder-kr/test.safetensors"
