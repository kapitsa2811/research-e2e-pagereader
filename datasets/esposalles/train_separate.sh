cd datasets/esposalles

./prepare.sh

cd ../..

CUDA_VISIBLE_DEVICES="0" python train.py --csv_train datasets/esposalles/train_htr.csv --csv_val datasets/esposalles/valid_htr.csv --csv_classes binary_class.csv --model_out esposalles_ 


CUDA_VISIBLE_DEVICES="0" python get_pagexmls.py --csv_val datasets/esposalles/test_htr.csv --csv_classes binary_class.csv --model trained_models/esposalles_csv_retinanet.pt

python dump_transcript_and_tag.py

cd ../NER-pytorch

CUDA_VISIBLE_DEVICES="0" python eval.py --test ../pytorch-retinanet/preds_txt --dir_out ../pytorch-retinanet/preds_csv

cd ../pytorch-retinanet/preds_csv

python ../evaluate2.py "Separate training"


