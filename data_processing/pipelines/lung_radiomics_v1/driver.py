import subprocess, click, json

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
def cli(cohort_id, container_id):
    subprocess.call(["python3","-m","data_processing.radiology.cli.load_dicom",            "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/load_dicom.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.window_dicom",          "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/load_dicom.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.generate_scan",         "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/generate_scan.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.randomize_contours",    "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/randomize_contours.json"])

    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_original_1.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_original_2.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_original_3.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_original_4.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_original_5.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_original_6.json"])

    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_pertubation_1.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_pertubation_2.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_pertubation_3.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_pertubation_4.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_pertubation_5.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.extract_radiomics",     "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/extract_radiomics_pertubation_6.json"])

    subprocess.call(["python3","-m","data_processing.radiology.cli.collect_csv_segment",   "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/collect_results_original_label.json"])
    subprocess.call(["python3","-m","data_processing.radiology.cli.collect_csv_segment",   "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/lung_radiomics_v1/collect_results_pertubation_label.json"])

if __name__ == "__main__":
    cli()
