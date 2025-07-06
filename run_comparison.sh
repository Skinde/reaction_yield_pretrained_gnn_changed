#python3 main_pretrain_pca.py --pca_dim 40 --seed 27407 
#python3 main_finetune_pca.py --data_id 1 --split_id 0 --train_size_id 0 > logs/finetune_pca_40features_1.log
#python3 main_finetune_pca.py --data_id 2 --split_id 0 --train_size_id 0 > logs/finetune_pca_40features_2.log
#python3 main_finetune_pca.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > logs/finetune_pca_40features_3.log


#python3 main_pretrain_pca.py --pca_dim 80 --seed 27407 
#python3 main_finetune_pca.py --data_id 1 --split_id 0 --train_size_id 0 > logs/finetune_pca_80features_800epoch_1.log
#python3 main_finetune_pca.py --data_id 2 --split_id 0 --train_size_id 0 > logs/finetune_pca_80features_800epoch_2.log
#python3 main_finetune_pca.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > logs/finetune_pca_80features_800epoch_3.log

python3 main_pretrain_umap.py --pca_dim 40 --seed 27407 

python3 main_finetune_umap.py --data_id 1 --split_id 0 --train_size_id 0 > logs/finetune_umap_40features_1.log
#python3 main_finetune_umap.py --data_id 2 --split_id 0 --train_size_id 0 > logs/finetune_umap_40features_2.log
#python3 main_finetune_umap.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > logs/finetune_umap_40features_3.log

#python3 main_pretrain_umap.py --pca_dim 80 --seed 27407 

#python3 main_finetune_umap.py --data_id 1 --split_id 0 --train_size_id 0 > logs/finetune_umap_80features_800epoch_1.log
#python3 main_finetune_umap.py --data_id 2 --split_id 0 --train_size_id 0 > logs/finetune_umap_80features_800epoch_2.log
#python3 main_finetune_umap.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > logs/finetune_umap_80features_800epoch_3.log

#python3 main_pretrain_tsne.py --pca_dim 3 --seed 27407 

#python3 main_finetune_tsne.py --data_id 1 --split_id 0 --train_size_id 0 > logs/finetune_tsne_3features_800epoch_1.log
#python3 main_finetune_tsne.py --data_id 2 --split_id 0 --train_size_id 0 > logs/finetune_tsne_3features_800epoch_2.log
#python3 main_finetune_tsne.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > logs/finetune_tsne_3features_800epoch_3.log

#python3 main_pretrain_isomap.py --pca_dim 40 --seed 27407 

#python3 main_finetune_isomap.py --data_id 1 --split_id 0 --train_size_id 0 > logs/finetune_isomap_40features_1.log
#python3 main_finetune_isomap.py --data_id 2 --split_id 0 --train_size_id 0 > logs/finetune_isomap_40features_2.log
#python3 main_finetune_isomap.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > logs/finetune_isomap_40features_3.log

#python3 main_pretrain_isomap.py --pca_dim 80 --seed 27407 

#python3 main_finetune_isomap.py --data_id 1 --split_id 0 --train_size_id 0 > logs/finetune_isomap_80features_800epoch_1.log
#python3 main_finetune_isomap.py --data_id 2 --split_id 0 --train_size_id 0 > logs/finetune_isomap_80features_800epoch_2.log
#python3 main_finetune_isomap.py --data_id 3 --split_id 1 --train_size_id 1 --seed 27407 > logs/finetune_isomap_80features_800epoch_3.log
