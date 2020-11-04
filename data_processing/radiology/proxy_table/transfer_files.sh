########################### setup ###########################

LOG_FILE=transfer_files.log
echo ">>>> writing data transfer logs to $LOG_FILE..."
echo "Running data_processing/radiology/proxy_table/transfer_files.sh with $1" >> $LOG_FILE;
exit_code=0

# TODO: validate data ingestion template

# set num_procs equal to magnitude of bandwidth.
# i.e. num_proc = bwlimit with last character stripped
bw = $BWLIMIT
num_procs = ${bw%?}

########################### transfer ###########################

# TODO: looks like the -h option can be removed, it is for help
# transfer whole files without delta-xfer algorithm from specified source location to destination location.
# spawn one process per unit of network bandwidth
# delete any files in the destination location that are not in the source location
# output stats at the end
# output a log file
# exclude transfer of any files with the excluded file extensions
# limit each process's network utilization to the specified bwlimit
time cat $CHUNK_FILE | xargs -I {} -P $num_procs -n 1 \
rsync -ahW --delete --stats --log-file=$LOG_FILE \
--exclude '*.'{$EXCLUDES} \
--bwlimit=$BWLIMIT  \
$HOST:$SOURCE_PATH/{} $DESTINATION_PATH

let exit_code = $? + $exit_code
echo "exit code after rsync = $exit_code" >> $LOG_FILE

########################### verify ###########################

# verify and log and log file counts
file_count = $(find mskmind_XNAT -type f -name "*" | wc -l)
[ $FILE_COUNT == $file_count ];

let exit_code = $? + $exit_code
echo "exit code after file_count verification = $exit_code" >> $LOG_FILE

# verify and log transfer data size (bytes)
data_size = $(find $DESTINATION_PATH -type f -name "*" | xargs du -ac)
[ $FILE_COUNT == $file_count ];

let exit_code = $?+$exit_code
echo "exit code after data_size verification = $exit_code" >> $LOG_FILE

# write manifest file
cp $1 $DESTINATION_PATH/manifest.yaml

let exit_code = $? + $exit_code
echo "exit code after write manifest file = $exit_code" >> $LOG_FILE

########################### teardown ###########################

# the end
if exit_code
then
        { echo "file transfer completed successfully!" ; exit 0 ; }
else
        { echo "file transfer failed. See $LOG_FILE" ; exit 1 ; }
fi
