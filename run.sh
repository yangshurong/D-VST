# bash run.sh \
#   train.py \
#   eval.py \
#   configs/training/train1.yaml
#   configs/eval/infer1.yaml
#   8 0,1,2,3,4,5,6,7
TRAINPY=$1
TRAIN_CONFIG=$2
NUM_GPU=$3
GPU_LIST=$4
export CUDA_VISIBLE_DEVICES=$GPU_LIST
PORT=$((RANDOM % 90 + 29510))
echo "PORT=\${PORT:-\"$PORT\"}"

/home/yangshurong/miniconda3/envs/py310/bin/accelerate launch \
    --config_file ./configs/accelerate_deepspeed.yaml \
    --main_process_port $PORT \
    --num_processes $NUM_GPU \
    $TRAINPY \
    --config $TRAIN_CONFIG
