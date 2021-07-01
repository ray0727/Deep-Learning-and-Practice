LOG=$(date +%m%d_%H%M_logs)
echo $LOG
mkdir $LOG
python3 train.py --y_condition --output_dir $LOG \
                  --batch_size 16 \
                  --epochs 50 \
                  --K 6 \
                  --L 3 \
                  # --saved_model 0614_2101_logs/glow_checkpoint_937.pt \
