import subprocess, click

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
def cli(cohort_id, container_id):
    subprocess.call(["python3","-m","data_processing.pathology.cli.infer_tile_labels",    "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/run_eng_inference/infer_tile_labels_resnet18.json"])
    subprocess.call(["python3","-m","data_processing.pathology.cli.visualize_tile_labels","-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/run_eng_inference/visualize_model_inference.json"])

if __name__ == "__main__":
    cli()

