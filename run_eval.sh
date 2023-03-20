ModelType=nodropPrototype-nodropRelation-lr-1e-5
#ModelType=nodropPrototype-nodropRelation-lr-5e-6

N=5
K=1

python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 --test test_wiki_input-$N-$K \
    --batch_size 4 --only_test \
    --load_ckpt ./checkpoint/$ModelType/camery-ready-$N-$K.pth.tar \
    --pretrain_ckpt ./data/bert-base-uncased \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model bert
