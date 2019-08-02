########## E2E NAMED ENTITY RECOGNITION ###################
: ' 
python train.py --csv_train ../DATASETS/gt-its/box_gt.csv\
        --csv_val ../DATASETS/gt-its/box_gt.csv\
        --csv_classes ../DATASETS/gt-its/classes.csv\
		--score_threshold 0.2 \
		--early_stop_crit cer \
		--train_htr True \
		--train_det True \
		--model_out its_ \
		--epochs_only_det 0 \
		--seg_level word\
        --pretrained_model trained_models/its_csv_retinanet.pt

'

########## PRETRAIN SYNTHETIC ###########

: '
python train.py --csv_train /home/mcarbonell/Documents/DATASETS/handwritten-document-synthesizer/synthesizer/box_ground_truth.txt\
                --csv_val /home/mcarbonell/Documents/DATASETS/handwritten-document-synthesizer/synthesizer/box_ground_truth.txt\
        --csv_classes ../DATASETS/gt-its/binary_class.csv\
		--score_threshold 0.3 \
		--early_stop_crit cer \
		--train_htr True \
		--train_det True \
		--model_out its_ \
		--epochs_only_det 0 \
        --epochs 4\
		--seg_level word
'
########## HTR AND DETECTION ###################
: '
python train.py --csv_train ../Documents/DATASETS/gt-its/box_gt_train.csv\
        --csv_val ../Documents/DATASETS/gt-its/box_gt_valid.csv\
        --csv_classes classes_its.csv\
		--score_threshold 0.5 \
        --max_iters_epoch 100\
		--early_stop_crit cer \
		--train_htr True \
		--train_det True \
		--model_out its_ \
		--epochs_only_det 1\
        --htr_gt_box True\
        --max_epochs_no_improvement 100\
        --depth 18\
		--seg_level word
        #--pretrained_model trained_models/synth_join_det_htr_ner_csv_retinanet.pt\
'

python3 train.py --csv_train datasets/esposalles/train_ner.csv\
        --csv_val datasets/esposalles/valid_ner.csv\
        --csv_classes datasets/esposalles/data-esposalles/classes.csv\
		--score_threshold 0.35 \
		--early_stop_crit cer \
        --epochs_only_det 2\
        --train_htr True \
		--train_det True \
        --model_out esposalles_det_\
        --max_epochs_no_improvement 200\
        --seg_level word\
        --max_boxes 600

