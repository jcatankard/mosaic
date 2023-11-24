# mosaic

[mosaic.streamlit.app](https://mosaic.streamlit.app/)

Streamlit app to create a photomosaic.

Tiles can fitted one of two ways:
1. finding the photo that matches closest each pixel of the target image subsection
     - this tends to yield a final image with more detail
2. finding the photo with the average colour that matches closest the average colour of the target picture subsection
     - this tends to yield a final image that matches the original image colours overall

Whether matching colour by pixel or subsection, given that each pixel is represented as three colours (RGB),
the image with the closest Euclidean distance is selected.

As well as choosing matching type, users can also choose the resolution of the tiles and overall image. 