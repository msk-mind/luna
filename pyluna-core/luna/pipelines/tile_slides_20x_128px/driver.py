import subprocess, click

@click.command()
@click.option('-s', '--container_id', required=True)
def cli(container_id):
    subprocess.call(["python3","-m","luna.pathology.cli.generate_tile_labels" ,"-s",container_id,"-m","luna/pipelines/tile_slides_20x_128px/generate_tile_labels_with_ov_labels.yml"])
    subprocess.call(["python3","-m","luna.pathology.cli.visualize_tile_labels","-s",container_id,"-m","luna/pipelines/tile_slides_20x_128px/visualize_tile_labels.yml"])
    subprocess.call(["python3","-m","luna.pathology.cli.save_tiles",           "-s",container_id,"-m","luna/pipelines/tile_slides_20x_128px/save_tiles.yaml"])
    subprocess.call(["python3","-m","luna.pathology.cli.collect_tile_segment", "-s",container_id,"-m","luna/pipelines/tile_slides_20x_128px/collect_tile_results.yml"])

if __name__ == "__main__":
    cli()

