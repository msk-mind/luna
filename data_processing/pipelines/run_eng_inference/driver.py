import subprocess, click

@click.command()
@click.option('-s', '--container_id', required=True)
def cli(container_id):
    subprocess.call(["python3","-m","data_processing.pathology.cli.infer_tile_labels",    "-s",container_id,"-m","data_processing/pipelines/run_eng_inference/infer_tile_labels_resnet18.yaml"])
    subprocess.call(["python3","-m","data_processing.pathology.cli.visualize_tile_labels","-s",container_id,"-m","data_processing/pipelines/run_eng_inference/visualize_model_inference.yaml"])

if __name__ == "__main__":
    cli()

