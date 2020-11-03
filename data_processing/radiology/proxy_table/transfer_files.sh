LOG_FILE=rsync.log
echo ">>>> writing data transfer logs to $LOG_FILE..."
echo "Running data_processing/radiology/proxy_table/transfer_files.sh with $1" >> $LOG_FILE;
exit_code=0

# verify data ingestion template


# set num_procs equal to magnitude of bandwidth.
# i.e. num_proc = bwlimit with last character stripped out
bw=$BWLIMIT
num_procs=${bw%?}

# transfer files
time cat $CHUNK_FILE | xargs -I {} -P $num_procs -n 1 \
rsync -ahW --delete --stats --log-file=$LOG_FILE \
--exclude '*.'{$EXCLUDES} \
--bwlimit=$BWLIMIT  \
$HOST:$SOURCE_PATH/{} $DESTINATION_PATH/

let exit_code=$?+$exit_code
echo "exit code after rsync = $exit_code" >> $LOG_FILE

# write manifest file
cp $1 $DESTINATION_PATH/manifest.yaml

let exit_code=$?+$exit_code
echo "exit code after write manifest file = $exit_code" >> $LOG_FILE

# verify transfer
    # verify and log exit codes for all processes above
    # verify and log transfer data size (bytes)
    # verify and log and log file counts
    # verify and log verification exit codes
