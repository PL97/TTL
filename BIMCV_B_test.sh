readonly DS=BIMCV
readonly EPOCH=200
readonly SUB=100
# readonly PRE=CheXpert
readonly PRE=imagenet
readonly TAR=all
readonly BS=128

python main.py --model resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc -1 --exp $1 --sub $SUB --target $TAR --test
python main.py --model resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 1 --exp $1 --sub $SUB --target $TAR --test
python main.py --model resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 2 --exp $1 --sub $SUB --target $TAR --test
python main.py --model resnet50 --bs $BS --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --trunc 3 --exp $1 --sub $SUB --target $TAR --test

python main.py --model slim_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 1 --exp $1 --sub $SUB --test
python main.py --model slim_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 2 --exp $1 --sub $SUB --test
python main.py --model slim_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --slim_from 3 --exp $1 --sub $SUB --test

python main.py --model freeze_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 1 --exp $1 --sub $SUB --test
python main.py --model freeze_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 2 --exp $1 --sub $SUB --test
python main.py --model freeze_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 3 --exp $1 --sub $SUB --test
python main.py --model freeze_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 4 --exp $1 --sub $SUB --test
python main.py --model freeze_resnet50 --bs 128 --data_parallel --num_workers 12 --max_epoch $EPOCH --pretrained $PRE --dataset $DS --finetune_from 5 --exp $1 --sub $SUB --test

