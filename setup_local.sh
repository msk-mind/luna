echo "Setting python paths with root dir=$PWD"
cp conf/logging.default.yml conf/logging.cfg
echo "Setting LUNA_HOME=$PWD"
export LUNA_HOME=$PWD
