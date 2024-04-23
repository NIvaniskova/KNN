import tensorflow_hub as hub
import tensorflow as tf
import os
import PIL.Image
import numpy as np
from dataclasses import dataclass
import numpy as np
from pathlib import Path

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def grayscale(img):
  # Turn image to almost greyscale, leaving 20% saturation.
  return tf.image.adjust_saturation(img, 0.2)


def add_directory_prefix(path: str) -> str:
  return 'transferred_' + path

# Directory with the original dataset.
dataset_dir = 'test_data'

# Directory with the style images.
style_dir = 'styles'

# Pairs of (style name w/o extension, style image)
styles = [ (Path(style).stem, load_img(os.path.join(style_dir, style))) 
            for style in os.listdir(style_dir)]

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Iterate over files in the directory.
# Requires the same structure as `WebFace`.
for sub_folder in os.scandir(dataset_dir):
  if sub_folder.is_dir():

    # Create an equivalent directory in the `transferred` directory.
    os.makedirs(add_directory_prefix(sub_folder.path), exist_ok=True)

    for file_name in os.scandir(sub_folder):
      if file_name.is_file():

        # Load the content image.
        content_image = grayscale(load_img(file_name.path))

        # Pick a random style.
        style = styles[np.random.randint(0, len(styles))]

        # Create the new file name.
        file_path, file_ext = os.path.splitext(file_name.path)
        new_file_name = add_directory_prefix(file_path + '_' + style[0] + file_ext)

        # Generate altered image.
        stylized_image = model(tf.constant(content_image), tf.constant(style[1]))[0]
        img = tensor_to_image(stylized_image)

        # Save the image.
        img.save(new_file_name)
