echo "_____Running annotation-regional ETLs_____"
echo "Running annotation-regional-refined-ovarian"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.dask_generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/OV_16-158/dask_regional_annotation_config_ov.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

echo "Running annotation-regional-refined-lung"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.dask_generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/LUNG_18-193/dask_regional_annotation_config_lung.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

echo "Running annotation-regional-refined-crc"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.dask_generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/CRC_21-167/dask_regional_annotation_config_crc.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

echo "Running annotation-regional-refined-br"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.dask_generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/BR_16-512/dask_regional_annotation_config_br.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

echo "_____Running annotation-point-proxy ETLs_____"
echo "Running annotation-point-proxy-ov ETL"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.proxy_table.generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/OV_16-158/point_js_ov.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

echo "Running annotation-point-proxy-br ETL"
time /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.proxy_table.generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/BR_16-512/point_js_br.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

echo "_____Running annotation-point-refined ETLs_____"
echo "Running annotation-point-refined-ov ETL"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.refined_table.generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/OV_16-158/point_geojson_ov.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

echo "Running annotation-point-refined-br ETL"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.refined_table.generate -d /gpfs/mskmindhdp_emc/etl-runner/configs/BR_16-512/point_geojson_br.yaml -a /gpfs/mskmindhdp_emc/etl-runner/configs/config.yaml

# echo "Update Graph"
# for filename in ../configs/*/graph-config-*; do
#     echo "Update graph with $filename"
#     /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.services.graph_service -d $filename -a ../configs/config.yaml
# done

# echo "DONE with nightly annotation pulls and graph updates."



