import subprocess, click

@click.command()
@click.option('-s', '--container_id', required=True)
def cli(container_id):
    subprocess.call(["python3","-m","luna.pathology.cli.infer_tile_labels",    "-s",container_id,"-m","luna/pipelines/run_eng_inference/infer_tile_labels_resnet18.yml"])
    subprocess.call(["python3","-m","luna.pathology.cli.visualize_tile_labels","-s",container_id,"-m","luna/pipelines/run_eng_inference/visualize_model_inference.yml"])

if __name__ == "__main__":
    cli()

