: 'python train.py --csv_train ../DATASETS/IAM/train_lines.csv \
		--csv_val ../DATASETS/IAM/valid_lines.csv \
		--score_threshold 0.35 \
		--early_stop_crit cer \
		--train_htr True \
		--train_det True \
		--model_out iam_join_det_htr_ \
		--epochs_only_det 5 \
		--seg_level line \
		--max_iters_epoch 100
'


 
python3 train.py --csv_train ../DATASETS/IAM/train_words.csv \
		--csv_val ../DATASETS/IAM/valid_words.csv \
		--score_threshold 0.1 \
		--early_stop_crit cer \
		--train_htr True \
		--train_det True \
		--model_out iam_join_det_htr_ \
		--epochs_only_det 0 \
		--seg_level word

#python test.py --model trained_models/iam_join_det_htr_csv_retinanet.pt --csv_val ../DATASETS/IAM/test_words.csv 

