cd /gpfs/mskmindhdp_emc/etl-runner/data-processing

echo "_____Running annotation-regional-proxy ETLs_____"
echo "Running annotation-regional-proxy-lung:"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.proxy_table.regional_annotation.generate -d data_processing/pathology/proxy_table/regional_annotation/proxy_bmp_data_config_lung.yaml -a data_processing/pathology/proxy_table/regional_annotation/config.yaml

echo "Running annotation-regional-proxy-ov:"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.proxy_table.regional_annotation.generate -d data_processing/pathology/proxy_table/regional_annotation/proxy_bmp_data_config_ov.yaml -a data_processing/pathology/proxy_table/regional_annotation/config.yaml

echo "_____Running annotation-regional-refined ETLs_____"

echo "Running annotation-regional-refined-lung"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d regional_geojson_lung.yaml -a config.yaml -p geojson
echo "Running annotation-regional-refined-ov"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d regional_geojson_ov.yaml -a config.yaml -p geojson


echo "_____Running annotation-regional-refined-concat ETLs_____"
echo "Running annotation-regional-refined-concat-lung"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d regional_concat_lung.yaml -a config.yaml -p concat

echo "Running annotation-regional-refined-concat-ov"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d regional_concat_ov.yaml -a config.yaml -p concat


echo "_____Running annotation-point-proxy ETLs_____"
echo "Running annotation-point-proxy-ov ETL"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.proxy_table.generate -d point_js_ov.yaml -a config.yaml


echo "_____Running annotation-point-refined ETLs_____"
echo "Running annotation-point-refined-ov ETL"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.refined_table.generate -d point_geojson_ov.yaml -a config.yaml

echo "Update Graph"
for filename in graph-config-*; do
    echo "Update graph with $filename"
    /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.services.graph_service -d $filename -a config.yaml
done

echo "DONE with nightly annotation pulls and graph updates."



