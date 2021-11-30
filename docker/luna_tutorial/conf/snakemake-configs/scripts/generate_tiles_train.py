slide_ids_train = ['2551571', '2551531', '2551028', '2551389']

import multiprocessing
import subprocess

root_path = "/gpfs/mskmindhdp_emc/user/shared_data_folder/pathology-tutorial/"

# simple wrapper around the cli for multiple slides
def pool_process(func, slides):
    pool = multiprocessing.Pool(3)
    pool.map(func, slides)
    pool.close()
    pool.join()

command = "generate_tiles -a " + root_path + "configs/snakemake-configs/app_config.yaml -s {slide} -m " + root_path + "configs/snakemake-configs/generate_tiles_train.yaml"
    
# call generate_tiles as subprocess
def call_generate_tiles(slide):
    subprocess.run(command, shell=True)
    return slide

pool_process(call_generate_tiles, slide_ids_train)