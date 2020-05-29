#!/bin/bash
# Basic while loop
# conda init bash
# source ~/.bashrc
conda activate pybnn
conda info
pwd
# export PATH=$PATH:`pwd`
# echo $PATH
counter=1

if [ $# -eq 0 ]
then
  cmax=10
  args=""
else
  cmax=$1
  if [ $# -eq 1 ]
  then
    args=""
  else
    args="--config $2"
    if [ $# -gt 2 ]
    then
      echo "Only two command line arguments allowed: number of iterations, configuration file. Extra
      arguments will be ignored."
    fi
  fi
fi

while [ $counter -le $cmax ]
do
  python bin/mlp_experiments.py `echo $args`
  ((counter++))
done
echo All done

