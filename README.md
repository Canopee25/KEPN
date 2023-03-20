### Environments
- ``python 3``
- ``PyTorch 1.7.1``
- ``transformers 4.6.0``


### Datasets and Models
You can find the training and validation data here: [FewRel 1.0 data](https://github.com/thunlp/FewRel/tree/master/data). For the test data, you can easily download from FewRel 1.0 competition website: https://codalab.lisn.upsaclay.fr/competitions/7395


### Code
Put all data in the **data** folder, CP pretrained model in the **CP_model** folder (you can download CP model from https://github.com/thunlp/RE-Context-or-Names/tree/master/pretrain or [Google Drive](https://drive.google.com/drive/folders/1AwQLqlHJHPuB1aKJ8XPHu8nu237kgtWj?usp=sharing)), and then you can simply use three scripts: *run_train.sh*, *run_eval.sh*, *run_submit.sh* for train, evaluation and test.


