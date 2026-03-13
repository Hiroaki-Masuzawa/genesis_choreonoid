set -x 
docker run -it --rm --gpus all  -v `pwd`:/userdir -w /userdir cpp_pred  bash