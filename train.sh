# PYTHONDONTWRITEBYTECODE=1 
export CUDA_VISIBLE_DEVICES=1
python train.py configs/ijepa/ijepa-adapter-base-upernet.py --amp 
# --work-dir work_dirs/ijepa-base_dpt/exp \
# python train.py configs/ijepa/ijepa-base_upernet.py --amp
      
