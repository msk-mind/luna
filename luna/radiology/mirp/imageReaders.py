
from luna.radiology.mirp.imageClass import ImageClass
from luna.radiology.mirp.roiClass import RoiClass
import SimpleITK as sitk
import numpy as np

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)

def read_itk_image(path_to_itk_file, modality=None):
    """ This takes a itk volume in, and spits out this "ImageClass thing """

    print ("Loading ", path_to_itk_file)

    # Load the image
    sitk_img = sitk.ReadImage(path_to_itk_file)

    # Import the image volume
    voxel_grid = sitk.GetArrayFromImage(sitk_img).astype(np.float)

    # Determine origin, spacing, and orientation
    image_origin        = np.array(sitk_img.GetOrigin())   [::-1]
    image_spacing       = np.array(sitk_img.GetSpacing())  [::-1]
    image_orientation   = np.array(sitk_img.GetDirection())[::-1]

    # Create an ImageClass object from the input image.
    image_obj = ImageClass(voxel_grid=voxel_grid,
                           origin=image_origin,
                           spacing=image_spacing,
                           orientation=image_orientation,
                           modality=modality,
                           spat_transform="base",
                           no_image=False)

    return image_obj

def read_itk_segmentation(path_to_seg_file):

    print ("Loading ", path_to_seg_file)

    # Load the segmentation file
    sitk_img = sitk.ReadImage(path_to_seg_file)

    # Obtain mask
    int_mask = sitk.GetArrayFromImage(sitk_img).astype(np.int)

    # Determine origin, spacing, and orientation
    mask_origin      = np.array(sitk_img.GetOrigin())   [::-1]
    mask_spacing     = np.array(sitk_img.GetSpacing())  [::-1]
    mask_orientation = np.array(sitk_img.GetDirection())[::-1]

    roi_list = []
   
    if len(np.unique(int_mask)) > 256:
        raise RuntimeWarning("More than 256 unique values in mask, perhaps this is an image, or something went really wrong?")

    for label_value in np.unique(int_mask):
        if label_value == 0: continue

        mask = np.where(int_mask==label_value, True, False)
        
        if mask.sum() < 5: continue # Accidental segmentations

        print ("Creating RoiClass for label value = ", label_value, mask.sum())

        # Create an ImageClass object using the mask.
        roi_mask_obj = ImageClass(voxel_grid=mask,
                                    origin=mask_origin,
                                    spacing=mask_spacing,
                                    orientation=mask_orientation,
                                    modality="SEG",
                                    spat_transform="base",
                                    no_image=False)

        roi_obj = RoiClass(name=f'labeled', contour=None, roi_mask=roi_mask_obj, label_value=label_value)
        roi_list.append(roi_obj)
 
    if len(roi_list) == 0: # Fixes 
        raise ValueError(f"Completely empty segmentation {path_to_seg_file}")

    return roi_list
