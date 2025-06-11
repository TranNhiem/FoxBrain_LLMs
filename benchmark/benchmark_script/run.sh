SRC_DIR=$(pwd)




python3 $SRC_DIR/generation_routine.py --config ./configs/tasks_config_translation_v5.json
python3 $SRC_DIR/generation_routine.py --config ./configs/tasks_config_v5.json
python3 $SRC_DIR/generation_routine.py --config ./configs/tmmlu_task_config_v5.json
python3 $SRC_DIR/generation_routine.py --config ./configs/tasks_config_cnn_v5.json
python3 $SRC_DIR/aggregate_results.py --result_dir ./outputs/v5 --model_name 'v5'