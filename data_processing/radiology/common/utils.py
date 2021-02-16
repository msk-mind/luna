import numpy as np
from PIL import Image
from medpy.io import load

def find_centroid(path, image_w, image_h):
    """
    Find the centroid of the 2d segmentation.

    :param path: filepath to 2d segmentation file
    :param image_w: width of the image
    :param image_h: height of the image
    :return: (x, y) center point
    """

    # 2d segmentation file path
    file_path = path.split(':')[-1]
    data, header = load(file_path)

    h, w, num_images = data.shape

    # Find the annotated slice
    xcenter, ycenter = 0, 0
    for i in range(num_images):
        seg = data[:,:,i]
        if np.any(seg):
            print(i)
            seg = seg.astype(float)

            # find centroid using mean
            xcenter = np.argmax(np.mean(seg, axis=1))
            ycenter = np.argmax(np.mean(seg, axis=0))
            break

    # Check if h,w matches IMAGE_WIDTH, IMAGE_HEIGHT. If not, this is due to png being rescaled. So scale centers.
    image_w, image_h = int(image_w), int(image_h)
    if not h == image_h:
        xcenter = int(xcenter * image_w // w)
    if not w == image_w:
        ycenter = int(ycenter * image_h // h)

    return (int(xcenter), int(ycenter))


def crop_images(xcenter, ycenter, dicom, overlay, crop_w, crop_h, image_w, image_h):
    """
    Crop PNG images around the centroid (xcenter, ycenter).

    :param xcenter: x center point to crop around. result of find_centroid()
    :param ycenter: y center point to crop around. result of find_centroid()
    :param dicom: dicom binary data
    :param overlay: overlay binary data
    :param crop_w: desired width of cropped image
    :param crop_h: desired height of the cropped image
    :param image_w: width of the original image
    :param image_h: height of the original image
    :return: binary tuple (dicom, overlay)
    """
    crop_w, crop_h = int(crop_w), int(crop_h)
    image_w, image_h = int(image_w), int(image_h)
    # Find xmin, ymin, xmax, ymax based on CROP_SIZE
    width_rad = crop_w // 2
    height_rad = crop_h // 2

    xmin, ymin, xmax, ymax = (xcenter - width_rad), (ycenter - height_rad), (xcenter + width_rad), (ycenter + height_rad)

    if xmin < 0:
        xmin = 0
        xmax = crop_w

    if xmax > image_w:
        xmin = image_w - crop_w
        xmax = image_w

    if ymin < 0:
        ymin = 0
        ymax = crop_h

    if ymax > image_h:
        ymin = image_h - crop_h
        ymax = image_h

    # Crop overlay, dicom pngs.
    dicom_img = Image.frombytes("L", (image_w, image_h), bytes(dicom))
    dicom_feature = dicom_img.crop((xmin, ymin, xmax, ymax)).tobytes()

    overlay_img = Image.frombytes("RGB", (image_w, image_h), bytes(overlay))
    overlay_feature = overlay_img.crop((xmin, ymin, xmax, ymax)).tobytes()

    return (dicom_feature, overlay_feature)


def dicom_to_bytes(dicom_path, width, height):
    """
    Create an image binary from dicom image.

    :param dicom_path: filepath to dicom
    :param width: width of the image
    :param height: height of the image
    :return: image in bytes
    """
    file_path = dicom_path.split(':')[-1]

    data, header = load(file_path)

    # Convert 2d image to float to avoid overflow or underflow losses.
    # Transpose to get the preserve x, y coordinates.
    image_2d = data[:,:,0].astype(float).T

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    im = Image.fromarray(image_2d_scaled)
    # resize pngs to user provided width/height
    im = im.resize( (width, height) )

    return im.tobytes()


def create_seg_images(src_path, uuid, width, height):
    """
    Create images from 3d segmentations.

    :param src_path: filepath to 3d segmentation
    :param uuid: scan uuid
    :param width: width of the image
    :param height: height of the image
    :return: an array of (instance_number, uuid, png binary) tuples
    """

    file_path = src_path.split(':')[-1]
    data, header = load(file_path)

    num_images = data.shape[2]

    # Find the annotated slices with 3d segmentation.
    # Some reverse engineering.. save the instance numbers
    # from the series to identify the dicom slices that were annotated.
    slices = []
    for i in range(num_images):
        image_slice = data[:,:,i]
        if np.any(image_slice):
            image_2d = image_slice.astype(float).T
            # double check that subtracting is needed for all.
            slice_num = num_images - (i+1)

            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            image_2d_scaled = np.uint8(image_2d_scaled)

            im = Image.fromarray(image_2d_scaled)
            # resize pngs to user provided width/height
            im = im.resize( (int(width), int(height)) )

            # save segmentation in red color.
            rgb = im.convert('RGB')
            red_channel = rgb.getdata(0)
            rgb.putdata(red_channel)
            png_binary = rgb.tobytes()

            slices.append( (slice_num, uuid, png_binary) )

    return slices


def overlay_images(dicom_path, seg, width, height):
    """
    Create dicom images.
    Create overlay images by blending dicom and segmentation images with 7:3 ratio.

    :param dicom_path: filepath to the dicom file
    :param seg: segmentation image in bytes
    :param width: width of the image
    :param height: height of the image
    :return: (dicom, overlay) tuple of binaries
    """
    width, height = int(width), int(height)
    dicom_binary = dicom_to_bytes(dicom_path, width, height)

    # load dicom and seg images from bytes
    dcm_img = Image.frombytes("L", (width, height), bytes(dicom_binary))
    dcm_img = dcm_img.convert("RGB")
    seg_img = Image.frombytes("RGB", (width, height), bytes(seg))

    res = Image.blend(dcm_img, seg_img, 0.3)
    overlay = res.tobytes()

    return (dicom_binary, overlay)
