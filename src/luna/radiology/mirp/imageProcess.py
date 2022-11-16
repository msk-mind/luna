import logging

from luna.radiology.mirp.imageClass import ImageClass

from skimage.segmentation import slic
from scipy.ndimage import binary_dilation, binary_erosion
import copy

import numpy as np
import pandas as pd


def saturate_image(img_obj, intensity_range, fill_value):

    # Sature image
    img_obj.saturate(intensity_range=intensity_range, fill_value=fill_value)

    return img_obj


def normalise_image(img_obj, norm_method, intensity_range=None, saturation_range=None, mask=None):

    if intensity_range is None:
        intensity_range = [np.nan, np.nan]

    if saturation_range is None:
        saturation_range = [np.nan, np.nan]

    # Normalise intensities
    img_obj.normalise_intensities(norm_method=norm_method,
                                  intensity_range=intensity_range,
                                  saturation_range=saturation_range,
                                  mask=mask)

    return img_obj


def resegmentise(img_obj, roi_list, settings):
    # Resegmentises segmentation map based on selected method

    if roi_list is not None:

        for ii in np.arange(0, len(roi_list)):

            # Generate intensity and morphology masks
            roi_list[ii].generate_masks()

            # Skip if no resegmentation method is used
            if settings.roi_resegment.method is None: continue

            # Re-segment image
            roi_list[ii].resegmentise_mask(img_obj=img_obj, by_slice=settings.general.by_slice, method=settings.roi_resegment.method, settings=settings)

            # Set the roi as the union of the intensity and morphological maps
            roi_list[ii].update_roi()

    return roi_list


def interpolate_image(img_obj, settings):
    # Interpolates an image set to a new spacing

    img_obj.interpolate(by_slice=settings.general.by_slice, settings=settings)

    return img_obj


def interpolate_roi(roi_list, img_obj, settings):
    # Interpolates roi to a new spacing
    for roi in roi_list:
        roi.interpolate(img_obj=img_obj, settings=settings)

    return roi_list


def estimate_image_noise(img_obj, settings, method="chang"):

    # TODO Implement as method for imageClass
    import scipy.ndimage as ndi

    # Skip if the image is missing
    if img_obj.is_missing:
        return -1.0

    if method == "rank":
        """ Estimate image noise level using the method by Rank, Lendl and Unbehauen, Estimation of image noise variance,
        IEEE Proc. Vis. Image Signal Process (1999) 146:80-84"""

        ################################################################################################################
        # Step 1: filter with a cascading difference filter to suppress original image volume
        ################################################################################################################

        diff_filter = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

        # Filter voxel volume
        filt_vox = ndi.convolve1d(img_obj.get_voxel_grid(), weights=diff_filter, axis=1)
        filt_vox = ndi.convolve1d(filt_vox, weights=diff_filter, axis=2)

        del diff_filter

        ################################################################################################################
        # Step 2: compute histogram of local standard deviation and calculate histogram
        ################################################################################################################

        # Calculate local means
        local_means = ndi.uniform_filter(filt_vox, size=[1, 3, 3])

        # Calculate local sum of squares
        sum_filter = np.array([1.0, 1.0, 1.0])
        local_sum_square = ndi.convolve1d(np.power(filt_vox, 2.0), weights=sum_filter, axis=1)
        local_sum_square = ndi.convolve1d(local_sum_square, weights=sum_filter, axis=2)

        # Calculate local variance
        local_variance = 1.0 / 8.0 * (local_sum_square - 9.0 * np.power(local_means, 2.0))

        del local_means, filt_vox, local_sum_square, sum_filter

        ################################################################################################################
        # Step 3: calculate median noise - this differs from the original
        ################################################################################################################

        # Set local variances below 0 (due to floating point rounding) to 0
        local_variance = np.ravel(local_variance)
        local_variance[local_variance < 0.0] = 0.0

        # Select robust range (within IQR)
        local_variance = local_variance[np.logical_and(local_variance >= np.percentile(local_variance, 25),
                                                       local_variance <= np.percentile(local_variance, 75))]

        # Calculate Gaussian noise
        est_noise = np.sqrt(np.mean(local_variance))

        del local_variance

    elif method == "ikeda":
        """ Estimate image noise level using a method by Ikeda, Makino, Imai et al., A method for estimating noise variance of CT image,
                Comp Med Imaging Graph (2010) 34:642-650"""

        ################################################################################################################
        # Step 1: filter with a cascading difference filter to suppress original image volume
        ################################################################################################################

        diff_filter = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

        # Filter voxel volume
        filt_vox = ndi.convolve1d(img_obj.get_voxel_grid(), weights=diff_filter, axis=1)
        filt_vox = ndi.convolve1d(filt_vox, weights=diff_filter, axis=2)

        ################################################################################################################
        # Step 2: calculate median noise
        ################################################################################################################

        est_noise = np.median(np.abs(filt_vox)) / 0.6754

        del filt_vox, diff_filter

    elif method == "chang":
        """ Noise estimation based on wavelets used in Chang, Yu and Vetterli, Adaptive wavelet thresholding for image
        denoising and compression. IEEE Trans Image Proc (2000) 9:1532-1546"""

        ################################################################################################################
        # Step 1: calculate HH subband of the wavelet transformation
        ################################################################################################################

        import pywt

        # Generate digital wavelet filter
        hi_filt = np.array(pywt.Wavelet("coif1").dec_hi)

        # Calculate HH subband image
        filt_vox = ndi.convolve1d(img_obj.get_voxel_grid(), weights=hi_filt, axis=1)
        filt_vox = ndi.convolve1d(filt_vox, weights=hi_filt, axis=2)

        ################################################################################################################
        # Step 2: calculate median noise
        ################################################################################################################

        est_noise = np.median(np.abs(filt_vox)) / 0.6754

        del filt_vox

    elif method == "immerkaer":
        """ Noise estimation based on laplacian filtering, described in Immerkaer, Fast noise variance estimation.
        Comput Vis Image Underst (1995) 64:300-302"""

        ################################################################################################################
        # Step 1: construct filter and filter voxel volume
        ################################################################################################################

        # Create filter
        noise_filt = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], ndmin=3)

        # Apply filter
        filt_vox = ndi.convolve(img_obj.get_voxel_grid(), weights=noise_filt)

        ################################################################################################################
        # Step 2: calculate noise level
        ################################################################################################################

        est_noise = np.sqrt(np.mean(np.power(filt_vox, 2.0))) / 36.0

        del filt_vox

    elif method == "zwanenburg":
        """ Noise estimation based on blob detection for weighting immerkaer filtering """

        ################################################################################################################
        # Step 1: construct laplacian filter and filter voxel volume
        ################################################################################################################

        # Create filter
        noise_filt = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], ndmin=3)

        # Apply filter
        filt_vox = ndi.convolve(img_obj.get_voxel_grid(), weights=noise_filt)
        filt_vox = np.power(filt_vox, 2.0)

        ################################################################################################################
        # Step 2: construct blob weighting
        ################################################################################################################

        # Spacing for gaussian
        gauss_filt_spacing = np.full(shape=(3), fill_value=np.min(img_obj.spacing))
        gauss_filt_spacing = np.divide(gauss_filt_spacing, img_obj.spacing)

        # Difference of gaussians
        weight_vox = ndi.gaussian_filter(img_obj.get_voxel_grid(), sigma=1.0 * gauss_filt_spacing) - ndi.gaussian_filter(img_obj.get_voxel_grid(), sigma=4.0 * gauss_filt_spacing)

        # Smooth edge detection
        weight_vox = ndi.gaussian_filter(np.abs(weight_vox), sigma=2.0*gauss_filt_spacing)

        # Convert to weighting scale
        weight_vox = 1.0 - weight_vox / np.max(weight_vox)

        # Decrease weight of vedge voxels
        weight_vox = np.power(weight_vox, 2.0)

        ################################################################################################################
        # Step 3: estimate noise level
        ################################################################################################################

        est_noise = np.sum(np.multiply(filt_vox, weight_vox)) / (36.0 * np.sum(weight_vox))
        est_noise = np.sqrt(est_noise)

    else:
        raise ValueError("The provided noise estimation method is not implemented. Use one of \"chang\" (default), \"rank\", \"ikeda\", \"immerkaer\" or \"zwanenburg\".")

    return est_noise


def get_supervoxels(img_obj, roi_obj, settings, n_segments=None):
    """Extracts supervoxels from an image"""

    # Check if image and/or roi exist, and skip otherwise
    if img_obj.is_missing or roi_obj.roi is None:
        print ("You tried to get supervoxel labels, however didn't pass in an image, so be careful!")
        return None

    # Get image object grid aauker: I don't think this is neccessary anymore
    img_voxel_grid = img_obj.get_voxel_grid()
    roi_bool_mask  = roi_obj.roi.get_voxel_grid().astype(np.bool)

    outside = binary_dilation(roi_bool_mask, iterations=int (10 // np.min(img_obj.spacing)))

    min_n_voxels = np.max([20.0, 100.0 / np.prod(img_obj.spacing)]) # Even smaller super voxels....
    segment_guess =  int(np.sum(outside) / min_n_voxels)
    print ("Starting guess: ", segment_guess)

    #inside  = binary_erosion(bool_mask,  iterations=10)
    #ring_mask = np.logical_and( outside, np.invert(inside))

    # Calculate roi pixel grid, low-level pixels, high-level pixels
    roi_level_pixels  = np.ma.array(img_voxel_grid, mask=np.invert(roi_bool_mask))
    low_level, high_level = roi_level_pixels.min() - 3 * roi_level_pixels.std(), roi_level_pixels.max() + 3 * roi_level_pixels.std()
    img_voxel_grid = np.clip( img_voxel_grid, low_level, high_level)

    # Convert to float with range [0.0, 1.0]
    img_voxel_grid = (img_voxel_grid - np.min(img_voxel_grid)) / np.ptp(img_voxel_grid)

    # Just also convert to floats
    img_voxel_grid = img_voxel_grid.astype(np.float)

    # Create a slic segmentation of the image stack
    img_segments = slic(image=img_voxel_grid, n_segments=segment_guess, spacing=img_obj.spacing, mask=outside, 
                       max_iter=50, sigma=1.0, compactness=0.04, multichannel=False, convert2lab=False, enforce_connectivity=True, start_label=1)

    return img_segments


def get_supervoxel_overlap(roi_obj, img_segments, mask=None):
    """Determines overlap of supervoxels with other the region of interest"""

    # Return None in case image segments and/or ROI are missing
    if img_segments is None or roi_obj.roi is None:
        return None, None, None

    # Check segments overlapping with the current contour
    if mask == "morphological" and roi_obj.roi_morphology is not None:
        overlap_segment_labels, overlap_size = np.unique(np.multiply(img_segments, roi_obj.roi_morphology.get_voxel_grid()), return_counts=True)
    elif mask == "intensity" and roi_obj.roi_intensity is not None:
        overlap_segment_labels, overlap_size = np.unique(np.multiply(img_segments, roi_obj.roi_intensity.get_voxel_grid()), return_counts=True)
    else:
        overlap_segment_labels, overlap_size = np.unique(np.multiply(img_segments, roi_obj.roi.get_voxel_grid()), return_counts=True)

    # Find super voxels with non-zero overlap with the roi
    overlap_size           = overlap_size[overlap_segment_labels > 0]
    overlap_segment_labels = overlap_segment_labels[overlap_segment_labels > 0]

    if len(overlap_size)==0: raise RuntimeError("No valid supervoxels found, this can happen if the entire grid recieved label 0 from slic, did you window out your tumor?")

    # Check the actual size of the segments overlapping with the current contour
    segment_size = list(map(lambda x: np.sum([img_segments == x]), overlap_segment_labels))

    # Calculate the fraction of overlap
    overlap_frac = overlap_size / segment_size

    return overlap_segment_labels, overlap_frac, overlap_size


def transform_images(img_obj, roi_list, settings, compute_features=False, extract_images=False, file_path=None):
    """
    Performs image transformations and calculates features.
    :param img_obj: image object
    :param roi_list: list of region of interest objects
    :param settings: configuration settings
    :param compute_features: flag to enable feature computation
    :param extract_images: flag to enable image exports
    :param file_path: path for image exports
    :return: list of features computed in the transformed image
    """

    # Empty list for storing features
    feat_list = []

    # Check if image transformation is required
    if not settings.img_transform.perform_img_transform:
        return feat_list

    # Get spatial filters to apply
    spatial_filter = settings.img_transform.spatial_filters

    # Iterate over spatial filters
    for curr_filter in spatial_filter:

        if curr_filter == "wavelet":
            # Wavelet filters
            from mirp.imageFilters.waveletFilter import WaveletFilter

            filter_obj = WaveletFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)

        elif curr_filter == "laplacian_of_gaussian":
            # Laplacian of Gaussian filters
            from mirp.imageFilters.laplacianOfGaussian import LaplacianOfGaussianFilter

            filter_obj = LaplacianOfGaussianFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)

        elif curr_filter == "laws":
            # Laws' kernels
            from mirp.imageFilters.lawsFilter import LawsFilter

            filter_obj = LawsFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)

        elif curr_filter == "mean":
            # Mean / uniform filter
            from mirp.imageFilters.meanFilter import MeanFilter

            filter_obj = MeanFilter(settings=settings)
            feat_list += filter_obj.apply_transformation(img_obj=img_obj, roi_list=roi_list, settings=settings,
                                                         compute_features=compute_features, extract_images=extract_images,
                                                         file_path=file_path)

        else:
            raise ValueError(f"{curr_filter} is not implemented as a spatial filter. Please use one of wavelet, laplacian_of_gaussian, mean or laws.")

    return feat_list


def crop_image(img_obj, roi_list=None, roi_obj=None, boundary=0.0, z_only=False):
    """ The function is used to slice a subsection of the image so that further processing is facilitated in terms of
     memory and computational requirements. """

    ####################################################################################################################
    # Initial steps
    ####################################################################################################################

    # Temporarily parse roi_obj to list, if roi_obj is provided and not roi_list. This is done for easier code maintenance.
    if roi_list is None:
        roi_list = [roi_obj]
        return_roi_obj = True
    else:
        return_roi_obj = False

    ####################################################################################################################
    # Determine region of interest bounding box
    ####################################################################################################################
    roi_ext_x = [];  roi_ext_y = []; roi_ext_z = []

    # Determine extent of all rois
    for roi_obj in roi_list:

        # Skip if the ROI is missing
        if roi_obj.roi is None:
            continue

        z_ind, y_ind, x_ind = np.where(roi_obj.roi.get_voxel_grid() > 0.0)

        # Skip if the ROI is empty
        if len(z_ind) == 0 or len(y_ind) == 0 or len(x_ind) == 0:
            continue

        roi_ext_z += [np.min(z_ind), np.max(z_ind)]
        roi_ext_y += [np.min(y_ind), np.max(y_ind)]
        roi_ext_x += [np.min(x_ind), np.max(x_ind)]

    # Check if the combined ROIs are empty
    if not (len(roi_ext_z) == 0 or len(roi_ext_y) == 0 or len(roi_ext_x) == 0):

        # Express mm boundary in voxels.
        boundary = np.ceil(boundary / img_obj.spacing).astype(np.int)

        # Concatenate extents for rois and add boundary to generate map extent
        ind_ext_z = np.array([np.min(roi_ext_z) - boundary[0], np.max(roi_ext_z) + boundary[0]])
        ind_ext_y = np.array([np.min(roi_ext_y) - boundary[1], np.max(roi_ext_y) + boundary[1]])
        ind_ext_x = np.array([np.min(roi_ext_x) - boundary[2], np.max(roi_ext_x) + boundary[2]])

        ####################################################################################################################
        # Resect image based on roi extent
        ####################################################################################################################

        img_res = img_obj.copy()
        img_res.crop(ind_ext_z=ind_ext_z, ind_ext_y=ind_ext_y, ind_ext_x=ind_ext_x, z_only=z_only)

        ####################################################################################################################
        # Resect rois based on roi extent
        ####################################################################################################################

        # Copy roi objects before resection
        roi_res_list = [roi_res_obj.copy() for roi_res_obj in roi_list]

        # Resect in place
        [roi_res_obj.crop(ind_ext_z=ind_ext_z, ind_ext_y=ind_ext_y, ind_ext_x=ind_ext_x, z_only=z_only) for roi_res_obj in roi_res_list]

    else:
        # This happens if all rois are empty - only copies of the original image object and the roi are returned
        img_res = img_obj.copy()
        roi_res_list = [roi_res_obj.copy() for roi_res_obj in roi_list]

    ####################################################################################################################
    # Return to calling function
    ####################################################################################################################
    
    if return_roi_obj:
        return img_res, roi_res_list[0]
    else:
        return img_res, roi_res_list


def interpolate_to_new_grid(orig_dim,
                            orig_spacing,
                            orig_vox,
                            sample_dim=None,
                            sample_spacing=None,
                            grid_origin=None,
                            translation=np.array([0.0, 0.0, 0.0]), order=1, mode="nearest", align_to_center=True, processor="scipy"):
    """
    Resamples input grid and returns the output grid.
    :param orig_dim: dimensions of the input grid
    :param orig_origin: origin (in world coordinates) of the input grid
    :param orig_spacing: spacing (in world measures) of the input grid
    :param orig_vox: input grid
    :param sample_dim: desired output size (determined within the function if None)
    :param sample_origin: desired output origin (in world coordinates; determined within the function if None)
    :param sample_spacing: desired sample spacing (in world measures; should be provided if sample_dim or sample_origin is None)
    :param translation: a translation vector that is used to shift the interpolation grid (in voxel measures)
    :param order: interpolation spline order (0=nnb, 1=linear, 2=order 2 spline, 3=cubic splice, max 5).
    :param mode: describes how to handle extrapolation beyond input grid.
    :param align_to_center: whether the input and output grids should be aligned by their centers (True) or their origins (False)
    :param processor: which function to use for interpolation: "scipy" for scipy.ndimage.map_coordinates and "sitk" for SimpleITK.ResampleImageFilter
    :return:
    """

    # Check if sample spacing is provided
    if sample_dim is None and sample_spacing is None:
        logging.error("Sample spacing is required for interpolation, but not provided.")

    # If no sample spacing is provided, assume original spacing. Note that for most purposes sample spacing should be provided
    if sample_spacing is None:
        sample_spacing = orig_spacing

    # Set sample spacing and orig_spacing to float
    sample_spacing = sample_spacing.astype(np.float)
    orig_spacing = orig_spacing.astype(np.float)

    # If no sample dimensions are provided, assume that the user wants to sample the original grid
    if sample_dim is None:
        sample_dim = np.ceil(np.multiply(orig_dim, orig_spacing / sample_spacing))

    # Set grid spacing (i.e. a fractional spacing in input voxel dimensions)
    grid_spacing = sample_spacing / orig_spacing

    # Set grid origin, if not provided previously
    if grid_origin is None:
        if align_to_center:
            grid_origin = 0.5 * (np.array(orig_dim) - 1.0) - 0.5 * (np.array(sample_dim) - 1.0) * grid_spacing

        else:
            grid_origin = np.array([0.0, 0.0, 0.0])

        # Update with translation vector
        grid_origin += translation * grid_spacing

    if processor == "scipy":
        import scipy.ndimage as ndi

        # Convert sample_spacing and sample_origin to normalised original spacing (where voxel distance is 1 in each direction)
        # This is required for the use of ndi.map_coordinates, which uses the original grid as reference.

        # Generate interpolation map grid
        map_z, map_y, map_x = np.mgrid[:sample_dim[0], :sample_dim[1], :sample_dim[2]]

        # Transform map to normalised original space
        map_z = map_z * grid_spacing[0] + grid_origin[0]
        map_z = map_z.astype(np.float32)
        map_y = map_y * grid_spacing[1] + grid_origin[1]
        map_y = map_y.astype(np.float32)
        map_x = map_x * grid_spacing[2] + grid_origin[2]
        map_x = map_x.astype(np.float32)

        # Interpolate orig_vox on interpolation grid
        map_vox = ndi.map_coordinates(input=orig_vox.astype(np.float32),
                                      coordinates=np.array([map_z, map_y, map_x], dtype=np.float32),
                                      order=order,
                                      mode=mode)

    elif processor == "sitk":
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)


        # Convert input voxel grid to sitk image. Note that SimpleITK expects x,y,z ordering, while we use z,y,
        # x ordering. Hence origins, spacings and sizes are inverted for both input image (sitk_orig_img) and
        # ResampleImageFilter objects.
        sitk_orig_img = sitk.GetImageFromArray(orig_vox.astype(np.float32), isVector=False)
        sitk_orig_img.SetOrigin(np.array([0.0, 0.0, 0.0]))
        sitk_orig_img.SetSpacing(np.array([1.0, 1.0, 1.0]))

        interpolator = sitk.ResampleImageFilter()

        # Set interpolator algorithm; SimpleITK has more interpolators, but for now use the older scheme for scipy.
        if order == 0:
            interpolator.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            interpolator.SetInterpolator(sitk.sitkLinear)
        elif order == 2:
            interpolator.SetInterpolator(sitk.sitkBSplineResamplerOrder2)
        elif order == 3:
            interpolator.SetInterpolator(sitk.sitkBSpline)

        # Set output origin and output spacing
        interpolator.SetOutputOrigin(grid_origin[::-1])
        interpolator.SetOutputSpacing(grid_spacing[::-1])
        interpolator.SetSize(sample_dim[::-1].astype(int).tolist())

        map_vox = sitk.GetArrayFromImage(interpolator.Execute(sitk_orig_img))
    else:
        raise ValueError("The selected processor should be one of \"scipy\" or \"sitk\"")

    # Return interpolated grid and spatial coordinates
    return sample_dim, sample_spacing, map_vox, grid_origin


def gaussian_preprocess_filter(orig_vox, orig_spacing, sample_spacing=None, param_beta=0.93, mode="nearest", by_slice=False):

    import scipy.ndimage

    # If no sample spacing is provided, assume original spacing. Note that for most purposes sample spacing should be provided
    if sample_spacing is None:
        sample_spacing = orig_spacing

    # Set sample spacing and orig_spacing to float
    sample_spacing = sample_spacing.astype(np.float)
    orig_spacing   = orig_spacing.astype(np.float)

    # Calculate the zoom factors
    map_spacing = sample_spacing / orig_spacing

    # Only apply to down-sampling (map_spacing > 1.0)
    # map_spacing[map_spacing<=1.0] = 0.0

    # Don't filter along slices if calculations are to occur within the slice only
    if by_slice: map_spacing[0] = 0.0

    # Calculate sigma
    sigma = np.sqrt(-8 * np.power(map_spacing, 2.0) * np.log(param_beta))

    # Apply filter
    new_vox = scipy.ndimage.gaussian_filter(input=orig_vox.astype(np.float32), sigma=sigma, order=0, mode=mode)

    return new_vox


def combine_pertubation_rois(roi_list, settings):
    new_roi_list = []
    for ii in np.arange(settings.vol_adapt.roi_random_rep): 
        roi_list_by_pertubation = [roi for roi in roi_list if roi.svx_randomisation_id == ii]
        repl_roi = roi_list_by_pertubation[0].copy()

        roi_vox_new = np.zeros(shape=repl_roi.roi.size, dtype=np.uint8)

        label_grid_mapping = { roi.label_value : roi.roi.get_voxel_grid() for roi in roi_list_by_pertubation}

        for label_value in sorted(label_grid_mapping.keys(), reverse=True):
            roi_vox_new[np.where(label_grid_mapping[label_value] != 0)] = label_value

        repl_roi.roi.set_voxel_grid(roi_vox_new)
        repl_roi.name += "_COMBINED"          # Adapt roi name
        new_roi_list += [repl_roi]
    return new_roi_list

def combine_all_rois(roi_list, settings):
    repl_roi = roi_list[0].copy()

    roi_vox_new = np.zeros(shape=repl_roi.roi.size, dtype=np.uint8)

    label_grid_mapping = { roi.label_value : roi.roi.get_voxel_grid() for roi in roi_list}

    for label_value in sorted(label_grid_mapping.keys(), reverse=True):
        roi_vox_new = np.where(label_grid_mapping[label_value] != 0, label_grid_mapping[label_value], roi_vox_new)
        # roi_vox_new[np.where(label_grid_mapping[label_value] != 0)] = label_grid_mapping[label_value]

    repl_roi.roi.set_voxel_grid(roi_vox_new)
    repl_roi.name += "_COMBINED"          # Adapt roi name
    return repl_roi



                
