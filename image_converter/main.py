from image_converter import ImageConverter


if __name__ == "__main__":
    ic = ImageConverter()
    directory = r"C:\Users\e10110199\Downloads\test photos"
    print("Number of files:", len(ic.find_files(directory)))

    ic.batch_convert(directory, new_type="JPEG")
