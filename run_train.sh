LR=1e-5
#LR=5e-6
#LR=9e-6
ModelType=nodropPrototype-nodropRelation-lr-$LR
N=5
K=5

# --load_ckpt ./checkpoint/$ModelType/camery-ready-$N-$K.pth.tar \


python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 --lr $LR \
    --pretrain_ckpt ./data/bert-base-uncased \
    --batch_size 4 --save_ckpt ./checkpoint/$ModelType/camery-ready-$N-$K.pth.tar \
    --cat_entity_rep \
    --backend_model bert
