# some examples

CUDA_VISIBLE_DEVICES=0 python scripts.py --ad_time 0 --data mnist --epsilon 0.1 --epoches 40 --arch 3conv
CUDA_VISIBLE_DEVICES=0 python scripts.py --ad_time 1 --data mnist --epsilon 0.1 --epoches 40 --arch 3conv
CUDA_VISIBLE_DEVICES=0 python scripts.py --ad_time 0 --data mnist --epsilon 0.2 --epoches 40 --arch 3conv
CUDA_VISIBLE_DEVICES=0 python scripts.py --ad_time 0 --data mnist --epsilon 0.1 --epoches 40 --arch 3conv --interval 1000
CUDA_VISIBLE_DEVICES=0 python scripts.py --ad_time 0 --data mnist --epsilon 0.1 --epoches 40 --arch resnet18
CUDA_VISIBLE_DEVICES=0 python scripts.py --ad_time 0 --data mnist --epsilon 0.1 --epoches 40 --arch resnet50
CUDA_VISIBLE_DEVICES=0 python scripts.py --ad_time 0 --data portraits --epsilon 0.031 --epoches 40 --arch 3conv