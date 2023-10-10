from typing import Optional
import requests
import string
import random
import subprocess
import zipfile
import os

def download_mp4(url: str, save_path: Optional[str] = None, max_seconds: Optional[int] = None) -> str:
    try:
        if save_path is None:
            N = 10
            name_file = ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))
            splited_url_list = url.split('.')
            print(splited_url_list)
            fpath = splited_url_list[-1] if len(splited_url_list) > 1 else 'mp4'
            print(fpath)
            save_path = f'input/{name_file}.{fpath}'
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for any errors in the response
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        if max_seconds:
            name_file = ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))
            splited_url_list = url.split('.')
            print(splited_url_list)
            fpath = splited_url_list[-1] if len(splited_url_list) > 1 else 'mp4'
            print(fpath)
            new_save_path = f'input/{name_file}.{fpath}'
            subprocess.check_output(['ffmpeg', '-i', save_path, '-t', str(max_seconds), '-c:v', 'copy', '-c:a', 'copy' , new_save_path], stderr=subprocess.STDOUT)
            os.remove(save_path)
            save_path = new_save_path
        print(f"Download complete! MP4 file saved at: {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the download: {e}")
        raise e
    

def download_and_extract_zip(url: str, extract_dir: str):
    try:
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()  # Check for download errors

        # Save the content to a temporary file
        with open("temp.zip", "wb") as f:
            f.write(response.content)

        # Extract the contents of the zip file to the specified directory
        with zipfile.ZipFile("temp.zip", "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Clean up: remove the temporary zip file
        os.remove("temp.zip")

        print("Download and extraction completed successfully.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
    except zipfile.BadZipFile as e:
        print(f"Failed to extract zip file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



def download_zip(url: str, zip_file: str = "temp.zip"):
    try:
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()  # Check for download errors

        # Save the content to a temporary file
        with open(zip_file, "wb") as f:
            f.write(response.content)
        return zip_file

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
    except zipfile.BadZipFile as e:
        print(f"Failed to extract zip file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_file(url: str, save_path: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for download errors
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully and saved as '{save_path}'.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")