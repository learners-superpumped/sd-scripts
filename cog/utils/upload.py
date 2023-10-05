import io

from typing import List
from gcloud.storage import Client
from PIL.Image import Image
from utils.logger import logger


def get_blob_uri(blob):
    return "gs://" + blob.id[: -(len(str(blob.generation)) + 1)]


def upload_images_to_gcs(
    images: List[Image], client: Client, bucket_name: str, dest: str
) -> None:
    result = []
    # Google Cloud Storage 클라이언트를 인증서와 함께 인스턴스화합니다.

    # GCS 버킷을 가져옵니다.
    bucket = client.get_bucket(bucket_name)

    # 이미지 리스트를 반복하며 각 이미지를 GCS에 업로드합니다.
    for i, image in enumerate(images):
        # 이미지 객체를 BytesIO 객체로 변환합니다.
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        # GCS에 업로드할 객체 이름을 설정합니다.
        object_name = f"inference-result/{dest}/{dest}-{i}.jpg"

        # 이미지 객체를 GCS에 업로드합니다.
        chunk_size = 256 * 1024  # 256KB
        try:
            blob = bucket.blob(object_name, chunk_size=chunk_size)
            blob.upload_from_file(image_bytes, content_type="image/jpeg")
        except Exception as e:
            logger.error(f"Failed to upload image to GCS: {e}")
        result.append(get_blob_uri(blob))

    print(f"{len(images)} images uploaded to {bucket_name}!")
    return result


def get_blob_uri(blob):
    return 'gs://' + blob.id[:-(len(str(blob.generation)) + 1)]


def upload_file(file_path: str, client: Client, bucket_name: str, model_type: str, request_id: str) -> str:
    bucket = client.get_bucket(bucket_name)
    upload_path = f"locon/{model_type}/{request_id}.safetensors"
    blob = bucket.blob(upload_path)
    blob.upload_from_filename(file_path)
    return get_blob_uri(blob)
