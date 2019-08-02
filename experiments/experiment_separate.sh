: '
echo "TRAIN DETECTION"
python train.py	--csv_train ../DATASETS/IAM/train_words.csv\
		--csv_val ../DATASETS/IAM/valid_words.csv\
		--score_threshold 0.35\
		--early_stop_crit map\
		--train_htr False\
		--train_det True\
		--model_out iam_detects_\
		--epochs_only_det 500\
		--max_epochs_no_improvement 50

'
echo "TRAIN RECOGNITION"
python train.py --csv_train ../DATASETS/IAM/train_words.csv \
		--csv_val ../DATASETS/IAM/valid_words.csv\
		--early_stop_crit cer \
		--train_htr True \
		--htr_gt_box True\
		--train_det False\
		--epochs_only_det 0\
		--model_out iam_transcribes_ 
		#--pretrained_model trained_models/iam_detects_csv_retinanet.pt\

: '
echo "GENERATE DETECTION PREDICTIONS ON TEST SET"
python get_pagexmls.py 	--csv_val ../DATASETS/IAM/test_words.csv \
			--model trained_models/iam_detects_csv_retinanet.pt\
			--score_threshold 0.35

python pagexml2csv.py 	--pxml_dir pagexmls\
			--fout predicted_boxes_test.csv

echo "CALCULATE CER ON TEST SET"
python test.py 	--model trained_models/iam_transcribes_csv_retinanet.pt\
		--csv_box_annot predicted_boxes_test.csv \
		--csv_val ../DATASETS/IAM/test_words.csv
'
