export DEVICE_NUM=8
export RANK_SIZE=8
export HCCL_WHITELIST_DISABLE=1
export GRAPH_OP_RUN=1
rm -rf ./train_parallel
mkdir ./train_parallel
cp -r ../deepxml ./train_parallel
cp -r ../*.py ./train_parallel
cp -r ../configure ./train_parallel
cd ./train_parallel || exit

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python ../../train.py --is_distributed True > logs.txt 2>&1 &