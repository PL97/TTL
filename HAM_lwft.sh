readonly DS=HAM
readonly EPOCH=200
readonly SUB=100
# readonly PRE=CheXpert
readonly PRE=imagenet
readonly TAR=all
readonly BS=256

# export CUDA_VISIBLE_DEVICES=1,2

python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 1 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 2 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 3 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 4 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 5 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 6 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 7 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 8 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 9 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 10 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 11 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 12 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 13 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 14 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 15 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 16 --exp $1 --sub $SUB --target $TAR
python main.py --model layer_wise_freeze_resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 17 --exp $1 --sub $SUB --target $TAR