cd datasets/esposalles

./prepare.sh

cd ../..

CUDA_VISIBLE_DEVICES="0" python train.py --csv_train datasets/esposalles/train_htr.csv --csv_val datasets/esposalles/valid_htr.csv --csv_classes binary_class.csv --model_out esposalles_ --epochs 2


CUDA_VISIBLE_DEVICES="0" python get_pagexmls.py --csv_val datasets/esposalles/test_htr.csv --csv_classes binary_class.csv --model trained_models/esposalles_csv_retinanet.pt

python dump_transcript_and_tag.py

cd ../NER-pytorch

python eval.py ../pytorch-retinanet/preds_csv




