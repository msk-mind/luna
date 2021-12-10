import multiprocessing
import subprocess

# ----------------------------------------------------------
# runs the load_slides cli

slide_ids = ['2551571', '2551531', '2551028', '2551389', '2551129']

root_path = "/gpfs/mskmindhdp_emc/user/shared_data_folder/pathology-tutorial/"

# simple wrapper around the cli for multiple slides
def pool_process(func, slides):
    pool = multiprocessing.Pool(3)
    pool.map(func, slides)
    pool.close()
    pool.join()
    
command = "load_slide -a " + root_path + "configs/snakemake-configs/app_config.yaml -s {slide} -m " + root_path + "configs/snakemake-configs/load_slides.yaml"

# call load_slide as subprocess
def call_load_slide(slide):
    subprocess.run(command, shell=True)
    return slide

pool_process(call_load_slide, slide_ids)