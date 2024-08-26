from pillow_heif import register_heif_opener
from PIL import Image
import logging
import os


VALID_TYPES = ["PNG", "JPG", "JPEG", "HEIC", "HEIF"]

logging.basicConfig(level=logging.INFO)


class ImageConverter:

    def __init__(self) -> None:
        # register heif and heic formats
        register_heif_opener()

    def batch_convert(self, directory: str, new_type: str = "JPEG") -> None:
        self._validate_file_type(new_type)
        self._validate_directory(directory)
        new_dir = self._create_directory(directory)
        files = self.find_files(directory)
        for i, file_name in enumerate(files, start=1):
            self._convert_file(directory, new_dir, file_name, new_type)
            logging.info(f"Converted: {'{:.0%}'.format(i / len(files))}")

    def image_convert(self, original_dir: str, new_type: str = "JPEG") -> None:
        pass

    def find_files(self, directory: str) -> list[str]:
        self._validate_directory(directory)
        files = [f for f in os.listdir(directory) if f.upper().split(".")[-1] in VALID_TYPES]
        if len(files) == 0:
            raise ValueError("No valid image files found.")
        return files

    @staticmethod
    def _create_directory(directory: str) -> str:
        new_dir = os.path.join(directory, "ConvertedFiles")
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    @staticmethod
    def _validate_file_type(file_type: str) -> None:
        if file_type.upper() not in VALID_TYPES:
            raise ValueError(f"{file_type} is not valid. Please select from {VALID_TYPES}.")

    @staticmethod
    def _validate_directory(directory: str) -> None:
        if not os.path.isdir(directory):
            raise ValueError(f"Directory '{directory}' does not exist.")

    @staticmethod
    def _convert_file(directory: str, new_dir: str, file_name: str, new_type: str) -> None:
        original_path = os.path.join(directory, file_name.upper())
        new_path = os.path.join(new_dir, os.path.splitext(file_name)[0] + "." + new_type.lower())
        new_type = "JPEG" if new_path.upper() == "JPG" else new_type.upper()

        try:
            with Image.open(original_path) as image:
                image.save(new_path, new_type)

        except (FileNotFoundError, OSError) as e:
            logging.error(f"Error converting {file_name}: {str(e)}")
