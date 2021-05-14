echo "_____Running annotation-regional ETLs_____"
echo "Running annotation-regional-refined-ovarian"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.dask_generate -d ../configs/OV_16-158/dask_regional_annotation_config_ov.yaml -a ../configs/config.yaml
echo "Running annotation-regional-refined-lung"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.dask_generate -d ../configs/LUNG_18-193/dask_regional_annotation_config_lung.yaml -a ../configs/config.yaml
echo "Running annotation-regional-refined-crc"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.dask_generate -d ../configs/CRC_21-167/dask_regional_annotation_config_crc.yaml -a ../configs/config.yaml


# echo "Update Graph"
# for filename in ../configs/*/graph-config-*; do
#     echo "Update graph with $filename"
#     /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.services.graph_service -d $filename -a ../configs/config.yaml
# done

# echo "DONE with nightly annotation pulls and graph updates."



