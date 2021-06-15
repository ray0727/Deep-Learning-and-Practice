LOG=$(date +%m%d_%H%M_logs)
echo $LOG
python3 train.py --y_condition --output_dir $LOG \
                  --batch_size 8 \
                  --epochs 25 \
                  --K 25 \
                  --L 3
