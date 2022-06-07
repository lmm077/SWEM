#!/bin/bash
uname -a
#env
date

GPU_IDS=0
MODEL='SWEM'
BACKBONE='resnet50'
STAGENAME='S3'

BATCHSIZE=8
NOBJ=2
NITER=4
KEYDIM=128
NBASES=128
TOPL=64

BACKEND='BACKBONE_'${BACKBONE}'_BS'${BATCHSIZE}'_NO'${NOBJ}'_KD'${KEYDIM}'_NI'${NITER}'_NB'${NBASES}'_TOP'${TOPL}
### main training
doTrain=true
doEval16=true
doEval17=true
#-m torch.distributed.launch --nproc_per_node=2
## 3 5*DAVIS + 1*YTVOS
if [ $doTrain = true ]; then
  CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 train.py \
  --model ${MODEL} \
  --backbone ${BACKBONE} \
  --key_dim ${KEYDIM} \
  --stage 3 \
  --stage_name ${STAGENAME} \
  --num_obj ${NOBJ} \
  --batch_size ${BATCHSIZE} \
  --lr 2e-5 \
  --em_iter ${NITER} \
  --num_bases ${NBASES} \
  --top_l ${TOPL} \
  --backend ${BACKEND}
fi

## evaluation
if [ $doEval16 = true ]; then
  EVALSET='DAVIS16'
  CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 eval.py \
  --eval_set ${EVALSET} \
  --model ${MODEL} \
  --backbone ${BACKBONE} \
  --key_dim ${KEYDIM} \
  --stage_name ${STAGENAME} \
  --num_obj ${NOBJ} \
  --em_iter ${NITER} \
  --num_bases ${NBASES} \
  --top_l ${TOPL} \
  --backend ${BACKEND}
fi

if [ $doEval17 = true ]; then
  EVALSET='DAVIS17'
  CUDA_VISIBLE_DEVICES=${GPU_IDS} python3 eval.py \
  --eval_set ${EVALSET} \
  --model ${MODEL} \
  --backbone ${BACKBONE} \
  --key_dim ${KEYDIM} \
  --stage_name ${STAGENAME} \
  --num_obj ${NOBJ} \
  --em_iter ${NITER} \
  --num_bases ${NBASES} \
  --top_l ${TOPL} \
  --backend ${BACKEND}
fi
