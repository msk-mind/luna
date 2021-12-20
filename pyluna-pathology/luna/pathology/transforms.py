from luna.pathology.common.utils import \
    get_layer_names, convert_xml_to_mask, get_stain_vectors_macenko, extract_patch_texture_features

from luna.pathology.common.preprocess import \
    get_downscaled_thumbnail, get_full_resolution_generator
import logging, os

import numpy as np
import pandas as pd

from PIL import Image

import openslide
import tifffile

from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

import itertools
from scipy import stats

class GenerateMaskImage:
    logger = logging.getLogger(__qualname__)
    def __init__(self, annotation_source='halo', annotation_name='Tumor', scale_factor=None):
        self.annotation_source = annotation_source
        self.scale_factor = scale_factor
        self.annotation_name = annotation_name

    def __call__(self, slide_path, roi_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        slide = openslide.OpenSlide(slide_path)
        slide.get_thumbnail((1000, 1000)).save(f"{output_dir}/slide_thumbnail.png")

        wsi_shape = slide.dimensions[1], slide.dimensions[0] # Annotation file has flipped dimensions w.r.t openslide conventions
        self.logger.info(f"Slide shape={wsi_shape}")

        layer_names     = get_layer_names(roi_path)
        self.logger.info(f"Available layer names={layer_names}")
        
        # x_pol, y_pol    = get_polygon_bounding_box(roi_path, self.annotation_name)
        # self.logger.info(f"Bounding box={x_pol}, {y_pol}")

        # x_roi, y_roi    = convert_halo_xml_to_roi(roi_path)

        mask_arr = convert_xml_to_mask(roi_path, wsi_shape, self.annotation_name)

        openslide.ImageSlide(Image.fromarray(255 * mask_arr)).get_thumbnail((1000, 1000)).save(f"{output_dir}/mask_thumbnail.png")

        tifffile.imsave(f"{output_dir}/mask_full_res.tif", mask_arr, compress=5)

        properties = {
            'mask_size': wsi_shape,
            'data': f"{output_dir}/mask_full_res.tif"
        }

        self.logger.info(properties)

        return properties


class TextureAnalysis:
    logger = logging.getLogger(__qualname__)
    def __init__(self, glcm_feature, tile_size=500, stain_sample_factor=10, stain_channel=0):
        self.tile_size = tile_size
        self.stain_sample_factor = stain_sample_factor
        self.stain_channel = stain_channel
        self.glcm_feature = glcm_feature
        self.scale_factor = None

    def __call__(self, slide_path, mask_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        slide = openslide.OpenSlide(slide_path)
        mask  = openslide.ImageSlide(mask_path)

        self.logger.info (f"Slide dimensions {slide.dimensions}, Mask dimensions {mask.dimensions}")

        sample_arr = get_downscaled_thumbnail(slide, self.stain_sample_factor)
        stain_vectors = get_stain_vectors_macenko(sample_arr)

        self.logger.info (f"Stain vectors={stain_vectors}")

        slide_full_generator, slide_full_level = get_full_resolution_generator(slide, tile_size=self.tile_size)
        mask_full_generator, mask_full_level   = get_full_resolution_generator(mask,  tile_size=self.tile_size)

        tile_x_count, tile_y_count = slide_full_generator.level_tiles[slide_full_level]
        self.logger.info("Tiles x %s, Tiles y %s", tile_x_count, tile_y_count)

        # populate address, coordinates
        address_raster = [address for address in itertools.product(range(tile_x_count), range(tile_y_count))]
        self.logger.info("Number of tiles in raster: %s", len(address_raster))

        features = []
        for address in tqdm(address_raster):
            mask_patch = np.array(mask_full_generator.get_tile(mask_full_level, address))[:, :, 0]

            if not np.count_nonzero(mask_patch) > 1: continue

            image_patch  = np.array(slide_full_generator.get_tile(slide_full_level, address))

            texture_values = extract_patch_texture_features(image_patch, mask_patch, stain_vectors, self.stain_channel, self.glcm_feature, plot=False)

            if not texture_values is None:
                features.append(texture_values)
                break
            
        features = np.concatenate(features).flatten()

        np.save(os.path.join(output_dir, "feature_vector.npy"), features)

        n, (smin, smax), sm, sv, ss, sk = stats.describe(features)
        ln_params = stats.lognorm.fit(features, floc=0)
   
        fx_name_prefix = f'pixel_original_glcm_{self.glcm_feature}_scale_{self.scale_factor}_channel_{self.stain_channel}'
        hist_features = {
            f'{fx_name_prefix}_nobs': n,
            f'{fx_name_prefix}_min': smin,
            f'{fx_name_prefix}_max': smax,
            f'{fx_name_prefix}_mean': sm,
            f'{fx_name_prefix}_variance': sv,
            f'{fx_name_prefix}_skewness': ss,
            f'{fx_name_prefix}_kurtosis': sk,
            f'{fx_name_prefix}_lognorm_fit_p0': ln_params[0],
            f'{fx_name_prefix}_lognorm_fit_p2': ln_params[2]
        }

        # The fit may fail sometimes, replace inf with 0
        df_result = pd.DataFrame(data=hist_features, index=[0]).replace([np.inf, -np.inf], 0.0).astype(float)
        self.logger.info (df_result)

        output_filename = os.path.join(output_dir, "stainomics.csv")

        df_result.to_csv(output_filename, index=False)

        properties = {
            'n_obs': n,
            'data': output_filename
        }

        return properties