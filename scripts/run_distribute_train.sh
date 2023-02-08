if [ $# != 2 ]
then
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DISTRIBUTE] [RANK_SIZE] "
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export RANK_SIZE=$2
export RANK_TABLE_FILE=$1
export GLOG_v=3
for((i=0;i<$RANK_SIZE;i++))
do
   export DEVICE_ID=$i
   rm -rf ./train_parallel$i
   mkdir ./train_parallel$i
   cp -r ../deepxml ./train_parallel$i
   cp -r ../*.py ./train_parallel$i
   cp -r ../configure ./train_parallel$i
   cd ./train_parallel$i || exit
   export RANK_ID=$i
   echo "start training for rank $RANK_ID, device $DEVICE_ID"
   env > env.log

   python train.py --is_distributed True > logs.txt 2>&1 &
   cd ..


done

