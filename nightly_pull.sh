cd /gpfs/mskmindhdp_emc/etl-runner/data-processing

echo "_____Running annotation-regional-proxy ETLs_____"
echo "Running annotation-regional-proxy-lung:"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.proxy_table.regional_annotation.generate -d ../configs/LUNG_18-193/proxy_bmp_data_config_lung.yaml -a ../configs/config.yaml

echo "Running annotation-regional-proxy-ov:"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.proxy_table.regional_annotation.generate -d ../configs/OV_16-158/proxy_bmp_data_config_ov.yaml -a ../configs/config.yaml

echo "Running annotation-regional-proxy-crc:"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.proxy_table.regional_annotation.generate -d ../configs/CRC_21-167/proxy_bmp_data_config_crc.yaml -a ../configs/config.yaml

echo "_____Running annotation-regional-refined ETLs_____"
echo "Running annotation-regional-refined-lung"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d ../configs/LUNG_18-193/regional_geojson_lung.yaml -a ../configs/config.yaml -p geojson
echo "Running annotation-regional-refined-ov"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d ../configs/OV_16-158/regional_geojson_ov.yaml -a ../configs/config.yaml -p geojson
echo "Running annotation-regional-refined-crc"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d ../configs/CRC_21-167/regional_geojson_crc.yaml -a ../configs/config.yaml -p geojson


echo "_____Running annotation-regional-refined-concat ETLs_____"
echo "Running annotation-regional-refined-concat-lung"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d ../configs/LUNG_18-193/regional_concat_lung.yaml -a ../configs/config.yaml -p concat

echo "Running annotation-regional-refined-concat-ov"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d ../configs/OV_16-158/regional_concat_ov.yaml -a ../configs/config.yaml -p concat

echo "Running annotation-regional-refined-concat-crc"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
        -d ../configs/CRC_21-167/regional_concat_crc.yaml -a ../configs/config.yaml -p concat


echo "_____Running annotation-point-proxy ETLs_____"
echo "Running annotation-point-proxy-ov ETL"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.proxy_table.generate -d ../configs/OV_16-158/point_js_ov.yaml -a ../configs/config.yaml


echo "_____Running annotation-point-refined ETLs_____"
echo "Running annotation-point-refined-ov ETL"
/gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.pathology.point_annotation.refined_table.generate -d ../configs/OV_16-158/point_geojson_ov.yaml -a ../configs/config.yaml

echo "Update Graph"
for filename in ../configs/*/graph-config-*; do
    echo "Update graph with $filename"
    /gpfs/mskmindhdp_emc/sw/env/bin/python3 -m data_processing.services.graph_service -d $filename -a ../configs/config.yaml
done

echo "DONE with nightly annotation pulls and graph updates."



