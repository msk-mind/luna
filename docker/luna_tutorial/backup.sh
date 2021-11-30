CONTAINER=$USER-luna-tutorial

# backup notebook
echo backing up notebooks
if [ -f notebooks/1_dataset-prep.ipynb ]
then
	mv notebooks/1_dataset-prep.ipynb notebooks/1_dataset-prep.ipynb.bk
fi
if [ -f notebooks/2_tiling.ipynb ]
then
	mv notebooks/2_tiling.ipynb notebooks/2_tiling.ipynb.bk
fi
if [ -f notebooks/3_model-training.ipynb ]
then
	mv notebooks/3_model-training.ipynb notebooks/3_model-training.ipynb.bk
fi
if [ -f notebooks/4_inference-and-visualization.ipynb ]
then
	mv notebooks/4_inference-and-visualization.ipynb notebooks/4_inference-and-visualization.ipynb.bk
fi
if [ -f notebooks/5_end-to-end-pipeline.ipynb ]
then
	mv notebooks/5_end-to-end-pipeline.ipynb notebooks/5_end-to-end-pipeline.ipynb.bk
fi
if [ -f notebooks/6_dsa-tools.ipynb ]
then
	mv notebooks/6_dsa-tools.ipynb notebooks/6_dsa-tools.ipynb.bk
fi        
if [ -f notebooks/7_teardown.ipynb ]
then
        mv notebooks/7_teardown.ipynb notebooks/7_teardown.ipynb.bk
fi

# copy notebooks from container
echo copying notebooks from container
docker cp $CONTAINER:/home/$USER/notebooks/1_dataset-prep.ipynb notebooks/1_dataset-prep.ipynb
docker cp $CONTAINER:/home/$USER/notebooks/2_tiling.ipynb notebooks/2_tiling.ipynb
docker cp $CONTAINER:/home/$USER/notebooks/3_model-training.ipynb notebooks/3_model-training.ipynb
docker cp $CONTAINER:/home/$USER/notebooks/4_inference-and-visualization.ipynb notebooks/4_inference-and-visualization.ipynb
docker cp $CONTAINER:/home/$USER/notebooks/5_end-to-end-pipeline.ipynb notebooks/5_end-to-end-pipeline.ipynb
docker cp $CONTAINER:/home/$USER/notebooks/6_dsa-tools.ipynb notebooks/6_dsa-tools.ipynb
docker cp $CONTAINER:/home/$USER/notebooks/7_teardown.ipynb notebooks/7_teardown.ipynb
        
# verify successful backup
echo verifying backups
if [ ! -f notebooks/1_dataset-prep.ipynb ]
then
	echo ERROR notebooks/1_dataset-prep.ipynb did not get backed up!
fi
if [ ! -f notebooks/2_tiling.ipynb ]
then
	echo ERROR notebooks/2_tiling.ipynb did not get backed up!
fi
if [ ! -f notebooks/3_model-training.ipynb ]
then
	echo ERROR notebooks/3_model-training.ipynb did not get backed up!
fi
if [ ! -f notebooks/4_inference-and-visualization.ipynb ]
then
	echo ERROR notebooks/4_inference-and-visualization.ipynb did not get backed up!
fi
if [ ! -f notebooks/5_end-to-end-pipeline.ipynb ]
then
	echo ERROR notebooks/5_end-to-end-pipeline.ipynb did not get backed up!
fi
if [ ! -f notebooks/6_dsa-tools.ipynb ]
then 
	echo ERROR notebooks/6_dsa-tools.ipynb did not get backed up!
fi
if [ ! -f notebooks/7_teardown.ipynb ]
then
        echo ERROR notebooks/7_teardown.ipynb did not get backed up!
fi
