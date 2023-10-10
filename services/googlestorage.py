from google.oauth2 import service_account
from google.cloud import storage
from random import choice
from uuid import uuid4


def generate_session_hash():
    return ''.join(choice('0123456789abcdef') for _ in range(30))


def upload_file(fname: str, bucket_name: str = "newtypev3", directory: str  = 'auto-generated-images', content_type: str = "image/png", use_random: bool = True) -> str:
    credentials = service_account.Credentials.from_service_account_file(
        "imagecreds.json")
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.get_bucket(bucket_name)
    fformat = fname.split('.')[-1]
    if use_random:
        random_string = str(uuid4())
        newfname = f'{random_string}.{fformat}'
    else:
        newfname = fname.split("/")[-1]
    blob = bucket.blob(f'{directory}/{newfname}')
    blob.content_type = content_type
    
    _ = blob.upload_from_filename(fname)
    blob.make_public()
    image_url = f'https://storage.googleapis.com/{bucket_name}/{directory}/{newfname}'
    return image_url