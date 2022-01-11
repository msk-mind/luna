# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('extract_stain_texture')

from luna.common.utils import cli_runner

_params_ = [('input_data', str), ('output_dir', str), ('repo_name', str), ('transform_name', str), ('model_name', str), ('weight_tag', str), ('num_cores', int), ('batch_size', int)]

@click.command()
@click.option('-i', '--input_data', required=False,
              help='path to input data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-rn', '--repo_name', required=False,
              help="repository name to pull model and weight from, e.g. msk-mind/luna-ml")
@click.option('-tn', '--transform_name', required=False,
              help="torch hub transform name")   
@click.option('-mn', '--model_name', required=False,
              help="torch hub model name")    
@click.option('-wt', '--weight_tag', required=False,
              help="weight tag filename")  
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-bx', '--batch_size', required=False,
              help="weight tag filename", default=256)    
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """ 

    """
    cli_runner( cli_kwargs, _params_, extract_stain_texture)


def extract_stain_texture():
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
