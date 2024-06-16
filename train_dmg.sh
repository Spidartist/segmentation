EXPORT CUDA_VISIBLE_DEVICES=1 
python train_damage.py --init_lr 5.0e-5 --warmup_epochs 1--num_epochs 20 --batchsize 2 --test_batchsize 2 --accum_iter 32 --type_damage ung_thu_da_day_20230620 configs/ijepa/ijepa-base-colonformer.py
python train_damage.py --init_lr 5.0e-5 --warmup_epochs 1--num_epochs 20 --batchsize 2 --test_batchsize 2 --accum_iter 32 --type_damage ung_thu_thuc_quan_20230620 configs/ijepa/ijepa-base-colonformer.py
python train_damage.py --init_lr 5.0e-5 --warmup_epochs 1--num_epochs 20 --batchsize 2 --test_batchsize 2 --accum_iter 32 --type_damage viem_da_day_20230620 configs/ijepa/ijepa-base-colonformer.py
python train_damage.py --init_lr 5.0e-5 --warmup_epochs 1--num_epochs 20 --batchsize 2 --test_batchsize 2 --accum_iter 32 --type_damage viem_thuc_quan_20230620 configs/ijepa/ijepa-base-colonformer.py
python train_damage.py --init_lr 5.0e-5 --warmup_epochs 1--num_epochs 20 --batchsize 2 --test_batchsize 2 --accum_iter 32 --type_damage viem_loet_hoanh_ta_trang_20230620 configs/ijepa/ijepa-base-colonformer.py
