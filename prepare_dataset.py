import mediapipe as mp
import argparse
import glob
import os
from PIL import Image, ImageOps
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='dataset/kr')
    parser.add_argument('--output_dir', type=str, default='dataset/cropped_kr')
    parser.add_argument('--max_crop_size', type=int, default=512)
    return parser.parse_args()


def crop_face(image: Image.Image, crop_height_rate: float = 1.3, crop_width_rate: float = 1.1):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    image = ImageOps.exif_transpose(image)
    w, h = image.size

    results = face_detection.process(np.array(image))

    if results.detections and len(results.detections) == 1:
        detection = results.detections[0]
        left = int(detection.location_data.relative_bounding_box.xmin * w)
        top = int(detection.location_data.relative_bounding_box.ymin * h)
        face_w = int(detection.location_data.relative_bounding_box.width * w)
        face_h = int(detection.location_data.relative_bounding_box.height * h)

        center_x = left + face_w // 2
        center_y = top + face_h // 2
        
        crop_h_size = int(face_h * crop_height_rate)
        crop_w_size = int(face_w * crop_width_rate)

        crop_left = max(0, center_x - crop_w_size // 2)
        crop_top = max(0, center_y - crop_h_size // 2)
        crop_right = min(w, crop_left + crop_w_size)
        crop_bottom = min(h, crop_top + crop_h_size)

        crop_x = crop_right - crop_left
        crop_y = crop_bottom - crop_top

        if crop_x > crop_y:
            crop_left += (crop_x - crop_y) // 2
            crop_right = crop_left + crop_y
        elif crop_x < crop_y:
            crop_top += (crop_y - crop_x) // 2
            crop_bottom = crop_top + crop_x

        face = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        center_x -= crop_left
        center_y -= crop_top
        return crop_left, crop_top, crop_right, crop_bottom, face 

    elif results.detections and len(results.detections) > 1:
        raise ValueError("Too many faces detected")
    else:
        raise ValueError("Face not detected")

def main(args):
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    image_paths = glob.glob(os.path.join(args.dataset_dir, '*'))

    for i, image_path in enumerate(image_paths):
        if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
            continue
        image = Image.open(image_path)
        try:
            crop_left, crop_top, crop_right, crop_bottom, face = crop_face(image)
        except ValueError:
            continue
        face = face.resize((args.max_crop_size, args.max_crop_size))
        face.save(os.path.join(args.output_dir, os.path.basename(image_path)))
        print(f'{i+1}/{len(image_paths)}: {image_path} cropped')


if __name__ == "__main__":
    args = parse_args()
    main(args)
