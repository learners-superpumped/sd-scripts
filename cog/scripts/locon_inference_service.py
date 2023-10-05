# parent path add to python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.locon_inference_service import LoconInferenceService
from schema.inference_request import IMG2IMGInferenceDTO, DDSDInferenceDTO, CoupleInferenceDTO, TXT2IMGInferenceDTO
import json
import argparse
import os
from utils.const import IMG2IMG, DDSD, COUPLE, TXT2IMG



def arg_parser():
    parser = argparse.ArgumentParser(description="Push messages to Redis queue")
    parser.add_argument(
        "--request_file",
        type=str,
        default="script/infer_request.json",
        help="Request file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save_dir",
        help="Directory to save the generated images",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=IMG2IMG,
        help="Type of inference",
    )
    return parser.parse_args()


def main(args):
    with open(args.request_file, "r") as file:
        inference_request = json.load(file)
    
    if args.type == IMG2IMG:
        dto = IMG2IMGInferenceDTO(**inference_request)
    elif args.type == DDSD:
        dto = DDSDInferenceDTO(**inference_request)
    elif args.type == COUPLE:
        dto = CoupleInferenceDTO(**inference_request)
    elif args.type == TXT2IMG:
        dto = TXT2IMGInferenceDTO(**inference_request)
    else:
        raise ValueError("Invalid inference type")

    locon_inference_service = LoconInferenceService()
    generated_images = locon_inference_service.predict(dto)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for i, image in enumerate(generated_images):
        filename = f"{i}.png"
        image.save(os.path.join(args.save_dir, filename))

    
if __name__ == "__main__":
    args = arg_parser()
    main(args)