python3 main_pretrain_umap.py --pca_dim 40 --seed 27407 > pretrain_umap.log

python3 main_finetune_umap.py --data_id 1 --split_id 0 --train_size_id 0 > finetune_4.log
python3 main_finetune_umap.py --data_id 2 --split_id 0 --train_size_id 0 > finetune_5.log
python3 main_finetune_umap.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > finetune_6.log
