import databricks.koalas as ks  

def process_patient(patient, target_spacing):
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """
    
    # TODO add multiprocess logging if spark performance doesn't work out.
    img_col = patient.img
    seg_col = patient.seg
    img_output = patient.preprocessed_img_path
    seg_output = patient.preprocessed_seg_path

    if os.path.exists(img_output) and os.path.exists(seg_output):
        logger.warning(img_output + " and " + seg_output + " already exists.")
        return

    if not os.path.exists(img_col):
        logger.warning(img_col + " does not exist.")
        return

    if not os.path.exists(seg_col):
        logger.warning(seg_col + " does not exist.")
        return

    img, img_header = load(img_col)
    target_shape = calculate_target_shape(img, img_header, target_spacing)

    img = resample_volume(img, 3, target_shape)
    np.save(img_output, img)
    logger.info("saved img at " + img_output)

    seg, _ = load(seg_col)
    seg = interpolate_segmentation_masks(seg, target_shape)
    np.save(seg_output, seg)
    logger.info("saved seg at " + seg_output)

    # since spark df is immutable, can't condition path columns (preprocessed_seg_path and preprocessed_img_path)
    # based on results of resampling volumes

    return

def process_patient_koalas_udf(patient):
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """
    # TODO add multiprocess logging if spark performance doesn't work out.
    img_col = patient.img.item()
    seg_col = patient.seg.item()

    img_output = patient.preprocessed_img_path.item()
    seg_output = patient.preprocessed_seg_path.item()
    target_spacing = (patient.preprocessed_target_spacing_x, patient.preprocessed_target_spacing_y, patient.preprocessed_target_spacing_z)
    
    if os.path.exists(img_output) and os.path.exists(seg_output):
        logger.warning(img_output + " and " + seg_output + " already exists.")
        return

    if not os.path.exists(img_col):
        logger.warning(img_col + " does not exist.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""
        return

    if not os.path.exists(seg_col):
        logger.warning(seg_col + " does not exist.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""
        return

    try: 
        img, img_header = load(img_col)
        target_shape = calculate_target_shape(img, img_header, target_spacing)

        img = resample_volume(img, 3, target_shape)
        np.save(img_output, img)
        logger.info("saved img at " + img_output)

        seg, _ = load(seg_col)
        seg = interpolate_segmentation_masks(seg, target_shape)
        np.save(seg_output, seg)
        logger.info("saved seg at " + seg_output)
    except:
        logger.warning("failed to generate resampled volume.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""

    return patient

def process_patient_koalas_udf_iterate_row(patient):
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """
    # TODO add multiprocess logging if spark performance doesn't work out.
    # for index, patient in df.iterrows():

    img_col = patient.img
    seg_col = patient.seg

    img_output = patient.preprocessed_img_path
    seg_output = patient.preprocessed_seg_path
    target_spacing = (patient.preprocessed_target_spacing_x, patient.preprocessed_target_spacing_y, patient.preprocessed_target_spacing_z)

    if os.path.exists(img_output) and os.path.exists(seg_output):
        logger.warning(img_output + " and " + seg_output + " already exists.")
        return

    if not os.path.exists(img_col):
        logger.warning(img_col + " does not exist.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""
        return

    if not os.path.exists(seg_col):
        logger.warning(seg_col + " does not exist.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""
        return

    try: 
        img, img_header = load(img_col)
        target_shape = calculate_target_shape(img, img_header, target_spacing)

        img = resample_volume(img, 3, target_shape)
        np.save(img_output, img)
        logger.info("saved img at " + img_output)

        seg, _ = load(seg_col)
        seg = interpolate_segmentation_masks(seg, target_shape)
        np.save(seg_output, seg)
        logger.info("saved seg at " + seg_output)
    except:
        logger.warning("failed to generate resampled volume.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""

    return patient


def process_patient_koalas_udf_iterate(df):
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """

    Parallel(n_jobs=8)(delayed(process_patient_koalas_udf_iterate_row)(patient) for _, patient in df.iterrows())
    return df


# Using Sparkdf + Parallel processing
results = Parallel(n_jobs=8)(delayed(process_patient)(row, target_spacing) for row in df.rdd.collect())

# Using Pandas DF and applyInPandas() [Apache Arrow] - best option:
print("** Testing Pandas UDF **")
df.groupBy("feature_uuid").applyInPandas(process_patient_pandas_udf, schema = df.schema).show()

# koalas - apply udf [single processing]
print("  ** Testing: Apply Koalas UDF **")
ks.set_option("compute.default_index_type", "distributed") 
df = df.to_koalas()         
df.groupby('feature_uuid').apply(process_patient_koalas_udf)
df = df.to_spark()

# koalas - batch apply udf - and PARALLEL processing on batches
print("  ** Testing: Koalas apply_batch udf and parallel processing on each batch **")
ks.set_option("compute.default_index_type", "distributed") 
df = df.to_koalas()         
df.koalas.apply_batch(process_patient_koalas_udf_iterate)
df = df.to_spark()