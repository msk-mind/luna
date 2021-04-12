import subprocess, click

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
def cli(cohort_id, container_id):
    subprocess.call(["python3","-m","data_processing.pathology.cli.generate_tile_labels" ,"-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/tile_slides_20x_128px/generate_tile_labels_with_ov_labels.json"])
    subprocess.call(["python3","-m","data_processing.pathology.cli.visualize_tile_labels","-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/tile_slides_20x_128px/visualize_tile_labels.json"])
    subprocess.call(["python3","-m","data_processing.pathology.cli.save_tiles",           "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/tile_slides_20x_128px/save_tiles.json"])
    subprocess.call(["python3","-m","data_processing.pathology.cli.collect_tile_segment", "-c",cohort_id,"-s",container_id,"-m","data_processing/pipelines/tile_slides_20x_128px/collect_tile_results.json"])

if __name__ == "__main__":
    cli()

