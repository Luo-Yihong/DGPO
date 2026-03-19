accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29502 scripts/train_sd3_dgpo.py --config config/dgpo_wocfg.py:general_ocr_sd3_4gpu
