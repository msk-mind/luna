class GeneralSettingsClass:

    def __init__(self):
        self.by_slice = None
        self.config_str = ""
        self.divide_disconnected_roi = "combine"
        self.no_approximation = False

class ImageInterpolationSettingsClass:


    def __init__(self):
        self.interpolate = True
        self.spline_order = 3
        self.new_spacing = [1.25, 1.25, 1.25]
        self.new_non_iso_spacing = [None]
        self.anti_aliasing = True
        self.smoothing_beta = 0.95
        self.bin = False
        self.bin_width = 10

class RoiInterpolationSettingsClass:

    def __init__(self):
        self.spline_order = 5
        self.incl_threshold = None

class ImagePerturbationSettingsClass:

    def __init__(self):

        self.roi_adapt_size = [0.0]
        self.roi_adapt_type = "distance"  # Alternatively, fraction for fractional volume growth and decreases
        self.rot_angles = [0.0]
        self.eroded_vol_fract = 0.8
        self.crop = True
        self.crop_distance = 150.0
        self.translate_frac = [0.0]
        self.add_noise = False
        self.noise_repetitions = 0
        self.noise_level = None
        self.randomise_roi = True
        self.roi_random_rep = 10
        self.drop_out_slice = 0

        # Division of roi into bulk and boundary
        self.bulk_min_vol_fract = 0.4
        self.roi_boundary_size = [0.0]

        # Selection of heterogeneous supervoxels
        self.heterogeneous_svx_count = 0.0
        self.heterogeneity_features  = ["rlm_sre"]
        self.heterogen_low_values    = [False]

        # Initially local variables
        self.translate_x = None
        self.translate_y = None

class Settings():
    def __init__(self):
        self.name = "MainSettings"
        self.vol_adapt = ImagePerturbationSettingsClass()
        self.general   = GeneralSettingsClass()
        self.img_interpolate = ImageInterpolationSettingsClass()
        self.roi_interpolate = RoiInterpolationSettingsClass()
