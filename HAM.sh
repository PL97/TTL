readonly DS=HAM
readonly EPOCH=200
readonly SUB=100
# readonly PRE=CheXpert
readonly PRE=imagenet
readonly TAR=all
readonly BS=256

# export CUDA_VISIBLE_DEVICES=1,2

# python main.py --model resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc -1 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 1 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 2 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 3 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 4 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 5 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 6 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 7 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 8 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 9 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 10 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 11 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 12 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 13 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 14 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 15 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 16 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 17 --exp $1 --sub $SUB --target $TAR



# python main.py --model resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc -1 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 1 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 2 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 3 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 4 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 5 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 6 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 7 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 8 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 9 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 10 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 11 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 12 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 13 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 14 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 15 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 16 --exp $1 --sub $SUB --target $TAR
# python main.py --model layerttl_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --pooling --trunc 17 --exp $1 --sub $SUB --target $TAR


# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 1 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 2 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 3 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 4 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 5 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 6 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 7 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 8 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 9 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 10 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 11 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 12 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 13 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 14 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 15 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 16 --exp $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 17 --exp $1 --sub $SUB --target $TAR


# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 1 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 2 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 3 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 4 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 5 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 6 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 7 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 8 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 9 --exp  $1 --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 10 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 11 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 12 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 13 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 14 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 15 --exp $1  --sub $SUB --target $TAR
# python main.py --model layer_wise_slim_resnet50 --bs $BS --pin_memory --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 16 --exp $1  --sub $SUB --target $TAR
