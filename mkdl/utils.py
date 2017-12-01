from PIL import ImageGrab, Image
import numpy as np


def prepare_image(im, conv_input_width, conv_input_height, conv_input_channels):
    im = im.resize((conv_input_width, conv_input_height))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((conv_input_height, conv_input_width, conv_input_channels))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr


def get_screenshot(screenshot_path):
    if screenshot_path == 'clip':
        return ImageGrab.grabclipboard()
    else:
        return Image.open(screenshot_path)
