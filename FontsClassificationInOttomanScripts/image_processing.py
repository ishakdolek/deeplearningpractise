from glob import glob
import os
import cv2
import uuid

from PIL import Image, ImageChops
from PIL import Image
import numpy as np


IMG_WIDTH = 500
IMG_HEIGHT = 50

path = "./fonts/data"

save_path = "./fonts/train"

filelist = glob(f"{path}/*/*.png")


def crop_surrounding_whitespace(image):
    """Remove surrounding empty space around an image.
    This implemenation assumes that the surrounding space has the same colour
    as the top leftmost pixel.
    :param image: PIL image
    :rtype: PIL image
    """
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if not bbox:
        return image
    return image.crop(bbox)


for filepath in filelist:

    filename = os.path.basename(filepath)

    if "nesih" in filepath:
        filename = f"{save_path}/nesih/{filename}"
    elif "rika" in filepath:
        filename = f"{save_path}/rika/{filename}"
    elif "talik" in filepath:
        filename = f"{save_path}/talik/{filename}"
    else:
        continue
    img = cv2.imread(filepath)
    image = Image.fromarray(img)
    image = crop_surrounding_whitespace(image)
    img = np.asarray(image)
    h, w, c = img.shape
    if w < 200:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dst = cv2.fastNlMeansDenoising(th2, 10, 10, 7)
    cv2.imwrite(filename, dst)
