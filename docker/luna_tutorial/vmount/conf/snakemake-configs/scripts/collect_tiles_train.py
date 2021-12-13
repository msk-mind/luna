slide_ids_train = ['2551571', '2551531', '2551028', '2551389']
root_path = "/gpfs/mskmindhdp_emc/user/shared_data_folder/pathology-tutorial/"

import multiprocessing
import subprocess

# simple wrapper around the cli for multiple slides
def pool_process(func, slides):
    pool = multiprocessing.Pool(3)
    pool.map(func, slides)
    pool.close()
    pool.join()
    
command = "collect_tiles -a " + root_path + "configs/snakemake-configs/app_config.yaml -s {slide} -m " + root_path + "configs/snakemake-configs/collect_tiles.yaml"

# call collect_tiles as subprocess
def call_collect_tiles(slide):
    subprocess.run(command, shell=True)

pool_process(call_collect_tiles, slide_ids_train)