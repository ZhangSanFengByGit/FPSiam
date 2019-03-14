if [ ! -d ./eval_output/ ] ; then
    mkdir eval_output
fi

model_name=V3_epoch_13
device0=0
device1=1
device2=2

CUDA_VISIBLE_DEVICES=$device0 nohup python eval.py --seqsIdx=0 --model=$model_name  &> ./eval_output/null0.out &
CUDA_VISIBLE_DEVICES=$device0 nohup python eval.py --seqsIdx=1 --model=$model_name  &> ./eval_output/null1.out &
CUDA_VISIBLE_DEVICES=$device0 nohup python eval.py --seqsIdx=2 --model=$model_name  &> ./eval_output/null2.out &
CUDA_VISIBLE_DEVICES=$device1 nohup python eval.py --seqsIdx=3 --model=$model_name  &> ./eval_output/null3.out &
CUDA_VISIBLE_DEVICES=$device1 nohup python eval.py --seqsIdx=4 --model=$model_name  &> ./eval_output/null4.out &
CUDA_VISIBLE_DEVICES=$device1 nohup python eval.py --seqsIdx=5 --model=$model_name  &> ./eval_output/null5.out &
CUDA_VISIBLE_DEVICES=$device1 nohup python eval.py --seqsIdx=6 --model=$model_name  &> ./eval_output/null6.out &
CUDA_VISIBLE_DEVICES=$device2 nohup python eval.py --seqsIdx=7 --model=$model_name  &> ./eval_output/null7.out &
CUDA_VISIBLE_DEVICES=$device2 nohup python eval.py --seqsIdx=8 --model=$model_name  &> ./eval_output/null8.out &
CUDA_VISIBLE_DEVICES=$device2 nohup python eval.py --seqsIdx=9 --model=$model_name  &> ./eval_output/null9.out &
CUDA_VISIBLE_DEVICES=$device2 nohup python eval.py --seqsIdx=10 --model=$model_name  &> ./eval_output/null10.out &
