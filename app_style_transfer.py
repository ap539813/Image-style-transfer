import functools

import altair as alt
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))



# @title Define image loading and visualization functions  { display-mode: "form" }

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[0], shape[1])
  offset_y = max(shape[0] - shape[1], 0) // 2
  offset_x = max(shape[1] - shape[0], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(uploaded_file, image_size=(256, 256), col = st):
  img = Image.open(uploaded_file)
  img = tf.convert_to_tensor(img)
  img = crop_center(img)
  img = tf.image.resize(img, image_size)
  if img.shape[-1] == 4:
        img = img[:, :, :3]
  img = tf.reshape(img, [-1, image_size[0], image_size[1], 3])/255
  col.image(np.array(img[0]))

  return img

def show_n(images, titles=('',), col = st):
  n = len(images)
  for i in range(n):
      col.image(np.array(images[i][0]))






## Basic setup and app layout
st.set_page_config(layout="wide")

alt.renderers.set_embed_options(scaleFactor=2)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



if __name__ == "__main__":
    img_width, img_height = 384, 384
    img_width_style, img_height_style = 384, 384
    col1, col2 = st.columns(2)
    col1.markdown('# Add image on which style is required')
    uploaded_file = col1.file_uploader(" Choose image to change")
    if uploaded_file is not None:
        content_image = load_image(uploaded_file, (img_width, img_height), col = col1)
     
    col2.markdown('# Add image from which style will be extracted')
    uploaded_file_style = col2.file_uploader("Choose style image")
    if uploaded_file_style is not None:
        style_image = load_image(uploaded_file_style, (img_width_style, img_height_style), col = col2)
        style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
        col3, col4, col5 = st.columns(3)
        col4.markdown('# Style applied on the image')
        show_n([stylized_image], titles=['Stylized image'], col = col4)