echo "Setting python paths with root dir=$PWD"
export PYTHONPATH=$PWD/pyluna-common:$PYTHONPATH
export PYTHONPATH=$PWD/pyluna-core:$PYTHONPATH
export PYTHONPATH=$PWD/pyluna-pathology:$PYTHONPATH
export PYTHONPATH=$PWD/pyluna-radiology:$PYTHONPATH
echo "Setting LUNA_HOME=$PWD"
export LUNA_HOME=$PWD
