import requests
from typing import Tuple, Optional
import numpy as np
import random
import string
import cv2


def read_image_from_url(url) -> np.ndarray:
    # Download the image using requests
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    # Convert the image content to a NumPy array
    image_array = np.frombuffer(response.content, dtype=np.uint8)

    # Read the image using cv2
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


def read_image_from_url_to_local(url, save_path: Optional[str] = None) -> Tuple[str, np.ndarray]:
    # Download the image using requests
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    # Convert the image content to a NumPy array
    image_array = np.frombuffer(response.content, dtype=np.uint8)

    # Read the image using cv2
    if not save_path:
        N = 10
        name_file = ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))
        splited_url_list = url.split('.')
        fpath = splited_url_list[-1] if len(splited_url_list) > 1 else 'png'
        save_path = f'input/{name_file}.{fpath}'

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite(save_path, image)
    return save_path, image