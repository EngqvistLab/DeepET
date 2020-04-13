#### Models

Models with Optimized hyper-parameters:
* `mpdels.ResNetN3()` + `Parameters.p_RES2_oriDist()`: the model architecture **type I** with otpimized hyperparams on datasets with a **originally distributed** dataset
* `models.RES1()` + 'Parameters.p_RES1_uniDist()': the model architecture **type I** with otpimized hyperparams	on datasets with a **uniformly distributed** dataset 
* `models.ResNetRed()` + `Parameters.p_ResNetRed_oriDist()`:the model architecture **type II** with otpimized hyperparams on datasets with a **originally distributed** dataset 
* `models.ResNetRed()` + `Parameters.p_ResNetRed_uniDist()`:the model architecture **type II** with otpimized hyperparams on datasets with a **originally distributed** dataset

#### All obove combinations were tested and the best model was obtained by 
```
cd ../scripts
python pre_train_simple.py \
--trainfile ../data/cleaned_ogts_train.fasta  \
--valfile ../data/cleaned_ogts_val.fasta  \
--modelname RES1 \
--hyparam p_RES1_uniDist \
--generator DataGenerator \
--mbatch 128 \
--lr 0.0001 \
--outdir ../results/pre_train_on_ogt_dataset/RES1_p_UniDist_B128_lr_1e-4

```

The rmse and r2_score of different experiments on validation and test datasets are 
* `../results/pre_train_on_ogt_dataset/pretrain_on_ogt_test_loss.pkl`
* `../results/pre_train_on_ogt_dataset/pretrain_on_ogt_val_loss.pkl`
