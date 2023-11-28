import mosaic
from PIL import Image

if __name__ == '__main__':
    target_src = None  # filepath to image
    folder = None  # 'filepath to folder
    tile_height = 16
    n_target_pixels = int(1e6)

    args = mosaic.prepare(n_target_pixels,
                          tile_height,
                          target_src=target_src,
                          tiles_src=folder
                          )
    results_array = mosaic.create(*args)
    Image.fromarray(results_array).show()
