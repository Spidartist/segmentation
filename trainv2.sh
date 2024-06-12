PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=1 \
python trainv2.py \
      --warmup_epochs 5 \
      --num_epochs 50 \
      --batchsize 8 \
      --test_batchsize 32 \
      --train_path /mnt/tuyenld/data/endoscopy/public_dataset/TrainDataset/ \
      configs/ijepa/ijepa-adapter-base-upernet-v2.py \
      --work-dir work_dirs/ijepa-adapter-base-upernet-v2 \