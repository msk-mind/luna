import logging

import numpy as np

from luna.radiology.mirp.imageProcess import crop_image, get_supervoxels, get_supervoxel_overlap
from luna.radiology.mirp.utilities import extract_roi_names


def rotate_image(img_obj, settings=None, rot_angle=None, roi_list=None):
    """ Rotation of image and rois """

    if settings is not None:
        rot_angle = settings.vol_adapt.rot_angles
    elif rot_angle is None:
        logging.error("No rotation angles were provided. A single rotation angle is expected.")

    if len(rot_angle) > 1:
        logging.warning("Multiple rotation angles were provided. Only the first is selected.")

    if type(rot_angle) is list:
        rot_angle = rot_angle[0]

    if rot_angle in [0.0, 360.0]:
        return img_obj, roi_list

    # Rotate rois
    if roi_list is not None:
        for ii in np.arange(0, len(roi_list)):
            roi_list[ii].rotate(angle=rot_angle, img_obj=img_obj)

    # Rotate image object
    img_obj.rotate(angle=rot_angle)

    return img_obj, roi_list


def randomise_roi_contours(img_obj, roi_list, settings):
    """Use SLIC to randomise the roi based on supervoxels"""

    # Check whether randomisation should take place
    if not settings.vol_adapt.randomise_roi:
        return roi_list

    from luna.radiology.mirp.utilities import world_to_index
    from scipy.ndimage  import binary_closing

    new_roi_list = []
    svx_roi_list = []

    # Iterate over roi objects
    for roi_ind in np.arange(0, len(roi_list)):
        print (f">>> Processing ROI with label [{roi_list[roi_ind].label_value}]")
        
        # Resect image to speed up segmentation process
        res_img_obj, res_roi_obj = crop_image(img_obj=img_obj, roi_obj=roi_list[roi_ind], boundary=5.0, z_only=False)

        print (f"Res_roi_obj shape = {res_roi_obj.roi.size}")
        # Calculate statistics on post-processed, cropped ROI
        res_roi_obj.calculate_roi_statistics(img_obj=res_img_obj, tag="postprocess")
            
        # tumor_volume     = res_roi_obj.roi.get_voxel_grid().sum() * np.prod(img_obj.spacing)
        # tumor_volume_1up = binary_dilation(res_roi_obj.roi.get_voxel_grid()).sum() * np.prod(img_obj.spacing)
        # tumor_surface_area = tumor_volume_1up-tumor_volume
        # print ("Volume, Differential Volume: ", tumor_volume, tumor_surface_area)

        # min_n_voxels = np.max([20.0, 250.0 / np.prod(res_img_obj.spacing)])
        # segment_guess =  int(np.prod(res_img_obj.size) / min_n_voxels)
        # print ("Starting guess: ", segment_guess)

        # for n_segments in np.linspace(segment_guess, segment_guess*5, 50):
        #     # Get supervoxels
        #     n_segments = int(n_segments)

        img_segments = get_supervoxels(img_obj=res_img_obj, roi_obj=res_roi_obj, settings=settings, n_segments=None)

        # Determine overlap of supervoxels with contour
        overlap_indices, overlap_fract, overlap_size = get_supervoxel_overlap(roi_obj=res_roi_obj, img_segments=img_segments)

        # Set the highest overlap to 1.0 to ensure selection of at least 1 supervoxel
        # aauker: aka, highest overlapping supervoxel is always included
        overlap_fract[np.argmax(overlap_fract)] = 1.0

        # Include supervoxels with 90% coverage and exclude those with less then 20% coverage
        a = 0.80
        b = 0.20
        
        overlap_fract[overlap_fract > a] = 1.0
        overlap_fract[overlap_fract < b] = 0.0

        candidate_indices  = overlap_indices[np.logical_and( overlap_fract > 0.0 , overlap_fract < 1.0 )]
        candidate_segments = np.where( np.isin(img_segments, candidate_indices), img_segments, 0 )

        average_segment_size = np.prod(img_obj.spacing) * np.where ( candidate_segments > 0, 1, 0).sum() / len(candidate_indices)
        
        print (f"Average segment size: {average_segment_size}")

            # if average_segment_size < 250: break

            # break # Use initial guess...for now

        print ("Candidate segments: ", len(candidate_indices))

        # Determine grid indices of the resected grid with respect to the original image grid
        grid_origin = world_to_index(coord=res_img_obj.origin, origin=img_obj.origin, spacing=img_obj.spacing, affine=img_obj.m_affine)
        grid_origin = grid_origin.astype(np.int)

        # Iteratively create randomised regions of interest
        for ii in np.arange(settings.vol_adapt.roi_random_rep):

            # Draw random numbers between 0.0 and 1.0
            random_incl = np.random.random(size=len(overlap_fract))

            # Select those segments where the random number is less than the overlap fraction - i.e. the fraction is the
            # probability of selecting the supervoxel
            incl_segments = overlap_indices[np.less(random_incl, overlap_fract)]

            # Replace randomised contour in original roi voxel space
            roi_vox = np.zeros(shape=roi_list[roi_ind].roi.size, dtype=np.bool)
            roi_vox[grid_origin[0]: grid_origin[0] + res_roi_obj.roi.size[0],
                    grid_origin[1]: grid_origin[1] + res_roi_obj.roi.size[1],
                    grid_origin[2]: grid_origin[2] + res_roi_obj.roi.size[2], ] = \
                np.reshape(np.in1d(np.ravel(img_segments), incl_segments),  res_roi_obj.roi.size)

            # Apply binary closing to close gaps
            roi_vox = binary_closing(input=roi_vox)

            # Update voxels in original roi, adapt name and set randomisation id

            repl_roi = roi_list[roi_ind].copy()
            repl_roi.roi.set_voxel_grid(voxel_grid=roi_vox)  # Replace copied original contour with randomised contour
            repl_roi.name += "_svx_" + str(ii)             # Adapt roi name
            repl_roi.svx_randomisation_id = ii         # Update randomisation id
            new_roi_list += [repl_roi]


        # Update voxels in original roi, adapt name and set randomisation id
        # Replace randomised contour in original roi voxel space
        roi_vox = np.zeros(shape=roi_list[roi_ind].roi.size, dtype=np.uint8)
        roi_vox[grid_origin[0]: grid_origin[0] + res_roi_obj.roi.size[0],
                grid_origin[1]: grid_origin[1] + res_roi_obj.roi.size[1],
                grid_origin[2]: grid_origin[2] + res_roi_obj.roi.size[2], ] = candidate_segments

        repl_roi = roi_list[roi_ind].copy()
        repl_roi.roi.set_voxel_grid(voxel_grid=roi_vox)  # Replace copied original contour with randomised contour
        repl_roi.name += "_SUPERVOXEL"         # Adapt roi name
        repl_roi.svx_randomisation_id = -1      # Update randomisation id
        svx_roi_list += [repl_roi]

    return new_roi_list, svx_roi_list


def adapt_roi_size(roi_list, settings):
    """ Adapt roi size by growing or shrinking the roi """

    # Adapt roi size by shrinking or increasing the roi
    new_roi_list = []

    # Get the adaptation size and type. Rois with adapt_size > 0.0 are dilated. Rois with adapt_size < 0.0 are eroded.
    # The type determines whether the roi is grown/shrunk with by certain distance ("distance") or to a certain volume fraction ("fraction")
    adapt_size_list = settings.vol_adapt.roi_adapt_size
    adapt_type      = settings.vol_adapt.roi_adapt_type

    # Iterate over roi objects in the roi list and adaptation sizes
    for roi_obj in roi_list:
        for adapt_size in adapt_size_list:
            if adapt_size > 0.0 and adapt_type == "distance":
                new_roi_obj  = roi_obj.copy()
                new_roi_obj.dilate(by_slice=settings.general.by_slice, dist=adapt_size)

                # Update name and adaptation size
                new_roi_obj.name += "_grow" + str(adapt_size)
                new_roi_obj.adapt_size = adapt_size

                # Add to roi list
                new_roi_list += [new_roi_obj]

            elif adapt_size < 0.0 and adapt_type == "distance":
                new_roi_obj = roi_obj.copy()
                new_roi_obj.erode(by_slice=settings.general.by_slice, dist=adapt_size, eroded_vol_fract=settings.vol_adapt.eroded_vol_fract)

                # Update name and adaptation size
                new_roi_obj.name += "_shrink" + str(np.abs(adapt_size))
                new_roi_obj.adapt_size = adapt_size

                # Add to roi list
                new_roi_list += [new_roi_obj]

            elif adapt_type == "fraction" and not adapt_size == 0.0:
                new_roi_obj = roi_obj.copy()
                new_roi_obj.adapt_volume(by_slice=settings.general.by_slice, vol_grow_fract=adapt_size)

                # Update name and adaptation size
                if adapt_size > 0:
                    new_roi_obj.name += "_grow" + str(adapt_size)
                else:
                    new_roi_obj.name += "_shrink" + str(np.abs(adapt_size))
                new_roi_obj.adapt_size = adapt_size

                # Add to roi list
                new_roi_list += [new_roi_obj]

            else:
                new_roi_list += [roi_obj]

    # Check for non-updated rois
    roi_names = extract_roi_names(new_roi_list)
    uniq_roi_names, uniq_index, uniq_counts = np.unique(np.asarray(roi_names), return_index=True, return_counts=True)
    if np.size(uniq_index) != len(roi_names):
        uniq_roi_list = [new_roi_list[ii] for ii in uniq_index]
    else:
        uniq_roi_list = new_roi_list

    # Return expanded roi list
    return uniq_roi_list
