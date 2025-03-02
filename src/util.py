import os
import requests
import subprocess
import zipfile
import tarfile
import shutil
from io import BytesIO
from pathlib import Path


def download_and_extract(
    urls, extract_to=Path("extracted_files"), unpack_top_level=False
):
    """
    Downloads and extracts all ZIP and TAR.GZ files from given URLs.

    :param urls: List of URLs of zip or tar.gz files.
    :param extract_to: Directory to extract files into (Path object).
    :param unpack_top_level: If True, moves the contents of the top-level directory inside the archive to extract_to.
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            content = BytesIO(response.content)

            # Try to extract as ZIP
            try:
                with zipfile.ZipFile(content) as archive:
                    print(f"Extracting ZIP: {url}")
                    temp_extract_path = extract_to / "temp_contents"
                    temp_extract_path.mkdir(parents=True, exist_ok=True)
                    archive.extractall(temp_extract_path)
            except zipfile.BadZipFile:
                # If not a ZIP, try as TAR.GZ
                try:
                    content.seek(0)  # Reset the stream position
                    with tarfile.open(fileobj=content, mode="r:gz") as archive:
                        print(f"Extracting TAR.GZ: {url}")
                        temp_extract_path = extract_to / "temp_contents"
                        temp_extract_path.mkdir(parents=True, exist_ok=True)
                        archive.extractall(temp_extract_path)
                except tarfile.TarError:
                    print(f"Unsupported file format: {url}")
                    continue

            # Handle top-level unpacking
            top_level_items = list(temp_extract_path.iterdir())
            if (
                unpack_top_level
                and len(top_level_items) == 1
                and top_level_items[0].is_dir()
            ):
                for item in top_level_items[0].iterdir():
                    item.rename(extract_to / item.name)
                top_level_items[0].rmdir()
            else:
                for item in top_level_items:
                    item.rename(extract_to / item.name)

            shutil.rmtree(temp_extract_path, ignore_errors=True)
            print(f"Extracted {url} to {extract_to}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")


def download_file(url: str, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    file_name = url.split("/")[-1]  # Extract filename from URL
    file_path = save_dir / file_name

    if file_path.exists():
        print(f"File already exists: {file_path}")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Downloaded: {file_path}")


def run_command(command: str):
    """
    Runs a shell command and streams both stdout and stderr in real-time.

    :param command: The shell command to execute.
    :raises Exception: If the command fails.
    """
    try:
        process = subprocess.Popen(
            ["/bin/bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffering
        )

        # Print output in real-time
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)

        process.wait()
        if process.returncode != 0:
            raise Exception("Command execution failed.")

    except Exception as e:
        print(f"Error: {str(e)}")
