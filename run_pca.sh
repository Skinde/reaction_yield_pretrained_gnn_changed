python3 main_pretrain_pca.py --pca_dim 40 --seed 27407 > pretrain_pca.log

python3 main_finetune_pca.py --data_id 1 --split_id 0 --train_size_id 0 > finetune_1.log
python3 main_finetune_pca.py --data_id 2 --split_id 0 --train_size_id 0 > finetune_2.log
python3 main_finetune_pca.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > finetune_3.log
