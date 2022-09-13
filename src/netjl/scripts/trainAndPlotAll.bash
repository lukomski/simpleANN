#!/bin/bash
set -e
DATETIME=`date +"%Y%m%d%H%M%S"`

EPOCHS=5

DATASETS=("Iris" "FashionMNIST" "DigitMNIST")

OUTPUTS_PAHT="src/netjl/outputs"

for DATASET in ${DATASETS[@]}; do
  {
    NAME="${DATETIME}_${DATASET}_${EPOCHS}"

    echo "====> Starting training dataset $NAME"
    julia src/netjl/main.jl -d ${DATASET} -e ${EPOCHS} -n ${NAME}

    echo "====> Starting plotter for $NAME"
	  julia src/common/plotter/plotter.jl -d ${NAME} -n netjl

    echo "====> Starting plotting confussion matrixes for $NAME"
    python src/netjl/scripts/plotConfussionMatrics.py \
      --metrics_file ${OUTPUTS_PAHT}/${NAME}/results/metrics.json \
      --out_files_directory_path ${OUTPUTS_PAHT}/${NAME}/results/confussion_matrixes/ \
      --config_file ${OUTPUTS_PAHT}/${NAME}/config.json
} &
done
echo "====> waiting"
wait
echo "====> done"
