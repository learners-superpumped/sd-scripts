import os
import mediapipe as mp
import numpy as np
from typing import List, Tuple
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import BaseOptions, vision

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.const import BACKGROUND, HAIR, FACE, BODY, CLOTHES, OTHERS
from utils.config import conf


class Segmentator:
    def __init__(self):
        self._ImageSegmenter = vision.ImageSegmenter
        ImageSegmenterOptions = vision.ImageSegmenterOptions
        VisionRunningMode = vision.RunningMode

        self._multiclass_segmentation_options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(
                conf.BASE_MODEL_PATH,
                'selfie_multiclass_256x256.tflite'
                )
            ),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True
        )

        self._interactive_segmentation_options = vision.InteractiveSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=os.path.join(conf.BASE_MODEL_PATH, 'magic_touch.tflite'),
            ),
        )

    def __call__(self, input_image: np.ndarray, categories: List[int], coord_rate: Tuple[float, float]) -> np.ndarray:
        with self._ImageSegmenter.create_from_options(self._multiclass_segmentation_options) as segmenter:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_image)
            segmentation_results = segmenter.segment(mp_image)
            category_mask = segmentation_results.category_mask
            category_mask = category_mask.numpy_view()
            category_mask = category_mask.copy()
            
            temp = 7
            for category in categories:
                if category not in [BACKGROUND, HAIR, FACE, BODY, CLOTHES, OTHERS]:
                    raise ValueError(f"Invalid category: {category}")
                category_mask[category_mask == category] = temp
            category_mask[category_mask != temp] = 0
            category_mask[category_mask == temp] = 1
            mask = category_mask.astype(np.uint8)

            one_person_mask = self._interactive_segmentation(input_image, coord_rate)
            mask = mask & one_person_mask
            
            return mask

    def _interactive_segmentation(self, input_image: np.ndarray, point: Tuple[int, int], threshold: float = 0.2) -> np.ndarray:
        x, y = point
        RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
        NormalizedKeypoint = containers.keypoint.NormalizedKeypoint
        with vision.InteractiveSegmenter.create_from_options(self._interactive_segmentation_options) as segmenter:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_image)
            roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x, y))
            segmentation_result = segmenter.segment(mp_image, roi)
            confidence_mask = segmentation_result.confidence_masks[0]
            confidence_mask = confidence_mask.numpy_view()
            confidence_mask = confidence_mask.copy()
            confidence_mask = confidence_mask > threshold
            mask = confidence_mask.astype(np.uint8)
            return mask


if __name__ == '__main__':
    import argparse
    import os
    import sys
    from PIL import Image
    # pythonpath
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--output_image', type=str, required=True)

    args = parser.parse_args()

    segmentator = Segmentator()
    input_image = np.array(Image.open(args.input_image))
    mask = segmentator(input_image, [FACE])

    h, w = input_image.shape[:2]
    print(h, w)

    one_person_mask = segmentator._interactive_segmentation(input_image, (0.5, 0.5))
    print(one_person_mask, np.mean(one_person_mask))
    mask = mask & one_person_mask
    # matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(mask)
    plt.savefig(args.output_image)