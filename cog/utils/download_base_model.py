import os
from gcloud import storage

import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()



def download_base_model(
    bucket: str,
    version: str,
    stage: str,
    model_name: str,
    base_model_path: str,
    gs_client: storage.Client
):
    """
    Download the base model from GCS and store it in the local cache.
    """
    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)
    if os.path.exists(f"{base_model_path}/{model_name}"):
        logger.info(f"Base model {model_name} already exists in {base_model_path}")
        return f"{base_model_path}/{model_name}"
    try:
        bucket = gs_client.get_bucket(bucket)
        blob = bucket.blob(f"{version}/{stage}/{model_name}")
        logger.info(f"Start downloding base model {model_name} from GCS to {base_model_path}/{model_name}")
        blob.download_to_filename(f"{base_model_path}/{model_name}")
        logger.info(f"Downloaded base model {model_name} from GCS to {base_model_path}/{model_name}")
        return f"{base_model_path}/{model_name}"
    except Exception as e:
        logger.error(f"Error downloading base model {model_name} from GCS: {e}")
        exit(1)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--bucket", type=str, default="stable-diffusion-model-repository-asia")
    args.add_argument("--version", type=str, default="v1")
    args.add_argument("--stage", type=str, default="production")
    args.add_argument("--model_name", type=str, default="")
    args.add_argument("--credentials", type=str, default="./env.json")
    args.add_argument("--base_model_path", type=str, default="./models")

    return args.parse_args()

if __name__ == "__main__":
    import json
    from oauth2client.service_account import ServiceAccountCredentials

    args = parse_args()

    # GCS Client
    with open(args.credentials, "r") as f:
        credentials_dict = json.load(f)

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict)
    gs_client = storage.Client(credentials=credentials, project="learneroid")
    download_base_model(
        bucket=args.bucket,
        version=args.version,
        stage=args.stage,
        model_name=args.model_name,
        base_model_path=args.base_model_path,
        gs_client=gs_client
    )