import numpy as np
from PIL import Image
from medpy.io import load
import cv2

def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize mr scan image intensity. Sets minimum value to zero, resacles by
    alpha factor and casts to uint8 w/ saturation.
    :param np.ndarray image: a single slice of an mr scan
    :return np.ndarray normalized_image: normalized mr slice
    """
    image = image - np.min(image)

    alpha_norm = 255.0 / min(np.max(image) - np.min(image), 10000)

    normalized_image = cv2.convertScaleAbs(image, alpha=alpha_norm)

    return normalized_image


def dicom_to_binary(dicom_path, width, height):
    """
    Create an image binary from dicom image.
    :param dicom_path: filepath to dicom
    :param width: width of the image
    :param height: height of the image
    :return: image
    """
    file_path = dicom_path.split(':')[-1]

    data, header = load(file_path)

    # Convert 2d image to float to avoid overflow or underflow losses.
    # Transpose to get the preserve x, y coordinates.
    image_2d = data[:,:,0].astype(float).T

    # Rescaling grey scale between 0-255
    image_2d_scaled = normalize(image_2d)

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    im = Image.fromarray(image_2d_scaled)
    # resize image to user provided width/height
    im = im.resize( (width, height) )

    return im.tobytes()