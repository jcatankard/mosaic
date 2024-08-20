from images import EXAMPLE_URL
from numpy.typing import NDArray
import streamlit as st
from PIL import Image
import mosaic
import io


def array_to_bytes(array: NDArray) -> bytes:
    img = Image.fromarray(array)
    with io.BytesIO() as byte_buffer:
        img.save(byte_buffer, format='JPEG')
        byte_values = byte_buffer.getvalue()
    return byte_values


def upload_target_image() -> None:
    help_text = 'Upload a png or jpg file'
    uploaded = st.file_uploader('Upload image', accept_multiple_files=False, type=['png', 'jpg'], help=help_text)
    st.session_state['target_src'] = uploaded.read() if uploaded is not None else EXAMPLE_URL
    st.image(st.session_state['target_src'])


def upload_tile_images() -> None:
    help_text = 'Upload png or jpg files'
    with st.form('tile-images-form', clear_on_submit=True):
        uploaded = st.file_uploader('Select images', accept_multiple_files=True, type=['png', 'jpg'], help=help_text)
        st.form_submit_button('Upload', type='primary')

    if len(uploaded) > 0:
        st.session_state['tile_src'] = [u.read() for u in uploaded]
        st.success(f"{len(st.session_state['tile_src'])} images uploaded", icon='✅')
    else:
        st.session_state['tile_src'] = None


def sidebar() -> None:
    with st.sidebar:
        st.header('Upload images')
        tabs = st.tabs(['Main image', 'Tile images'])
        with tabs[0]:
            upload_target_image()
        with tabs[1]:
            upload_tile_images()


def select_pixel_variables() -> None:
    help_text = 'Represents the number of pixels along the edge of each tile'
    st.session_state['tile_height'] = st.slider('Select tile size', 4, 32, 16, help=help_text)

    help_text = 'The larger the output, the more pixels in the final image and the more tiles that can fit inside'
    target_size = st.slider('Select output size', 1, 100, 14, help=help_text)
    st.session_state['target_pixels'] = target_size * 10 ** 5


def create() -> None:
    if st.button('Create', type='primary'):
        args = mosaic.prepare(
            st.session_state['target_pixels'],
            st.session_state['tile_height'],
            st.session_state['target_src'],
            st.session_state['tile_src']
        )
        st.session_state['results_array'] = mosaic.create(*args)


def app() -> None:
    st.set_page_config(layout='centered', page_title='Mosaic', page_icon='📷')
    st.title('Mosaic')
    select_pixel_variables()
    sidebar()
    create()
    if 'results_array' in st.session_state:
        byte_values = array_to_bytes(st.session_state['results_array'])
        st.download_button('Download mosaic', file_name='My mosaic.png', data=byte_values)
        st.image(byte_values, use_column_width=True)


if __name__ == '__main__':
    app()
