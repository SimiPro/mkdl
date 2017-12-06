from PIL import ImageGrab, Image
import numpy as np

INPUT_WIDTH = 200
INPUT_HEIGHT = 66
INPUT_CHANNELS = 3

old_image = None


def prepare_image(im):
    return prepare_image_(im, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)


def prepare_image_(im, conv_input_width, conv_input_height, conv_input_channels):
    im = im.resize((conv_input_width, conv_input_height))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((conv_input_height, conv_input_width, conv_input_channels))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr


def get_screenshot(screenshot_path):
    im = None
    if screenshot_path == 'clip':
        im = ImageGrab.grabclipboard()
    else:
        im = Image.open(screenshot_path)
    if im is not None:
        global old_image
        old_image = im
        return im
    print('had to take old image!! failed to take screenshot!')
    return old_image


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)