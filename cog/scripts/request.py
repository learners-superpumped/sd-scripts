import requests
import time

url = 'http://localhost:5001/predictions'

prompt = "leogirl, hoge person, realistic Documentary photography, detailed face cleavage, realistic, photorealistic"
negative_prompt = "(worst quality, low quality, cgi, bad eye, worst eye, illustration, cartoon), deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, open mouth"
guidance_scale = 7.5
num_inference_steps = 30

start = time.time()
response = requests.post(
    url,
    json={
        "input":{
            "image": "https://ynoblesse.com/wp-content/uploads/2023/04/339838208_917450576129267_1358914827833185623_n.jpg",
            "locon_url": "https://replicate.delivery/pbxt/yPm4ParArO6wL9rc5l6xM1laVB4LojivMfFIMxErYZGCuatIA/locon.safetensors",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }
    }
).json()
print(response)
print(time.time() - start)