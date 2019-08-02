git lfs clone  git@bitbucket.org:qidenus/data-esposalles.git

python data-esposalles/code/add_valid_partition.py


cd data-esposalles/train
python ../code/add_record_coords.py
python ../code/create_cropped_records_ds.py ../../train_htr.csv
cd ../..

cd data-esposalles/valid
python ../code/add_record_coords.py
python ../code/create_cropped_records_ds.py ../../valid_htr.csv

cd ../..


cd data-esposalles/test
python ../code/add_record_coords.py
python ../code/create_cropped_records_ds.py ../../test_htr.csv

cd ../..
cd ../..

python get_transcript_gt.py datasets/esposalles/train_htr.csv
python get_transcript_gt.py datasets/esposalles/valid_htr.csv
python get_transcript_gt.py datasets/esposalles/test_htr.csv

