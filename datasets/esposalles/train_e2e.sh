cd datasets/esposalles

./prepare_basic_e2e.sh

cd ../..

CUDA_VISIBLE_DEVICES="1" python train.py --csv_train datasets/esposalles/train_basic_e2e.csv --csv_val datasets/esposalles/valid_basic_e2e.csv --csv_classes datasets/esposalles/classes.csv --model_out esposalles_e2e_basic_

CUDA_VISIBLE_DEVICES="1" python get_pagexmls.py --csv_val datasets/esposalles/test_basic_e2e.csv --csv_classes datasets/esposalles/classes.csv --model trained_models/esposalles_e2e_basic_csv_retinanet.pt

python create_csv_iehhr.py

cd preds_csv_iehhr

python ../evaluate2.py 'e2e train'
