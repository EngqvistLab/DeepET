#### split the topt dataset into train and test datasets
`split_train_test_topt_datasets.ipynb`
  
#### Test transfer learning strategies
```
cd ../../scripts/
```

* Fine-tune only dense layers:
```
for i in {1..10}
do
python TLFT.py \
--modelfile ../results/pre_train_on_ogt_dataset/RES1_UniDist_D2512_1e4_B128/bestmodel.h5  \
--trainfile ../transfer_learning/Topt/data/cleaned_enzyme_topts_v1_train.fasta \
--testfile ../transfer_learning/Topt/data/cleaned_enzyme_topts_v1_test.fasta \
--hyparam p_RES1_uniDist_best1 \
--layer flatten_1 \
--tag $i \
--savemodel Y \
--outdir ../results/topt_models/TL6_RES1_ogt_onehot_SameAnces

done
```

* Fine-tune all layers:
```
for i in {1..10}
do
python TLFT.py \
--modelfile ../results/pre_train_on_ogt_dataset/RES1_UniDist_D2512_1e4_B128/bestmodel.h5  \
--trainfile ../transfer_learning/Topt/data/cleaned_enzyme_topts_v1_train.fasta \
--testfile ../transfer_learning/Topt/data/cleaned_enzyme_topts_v1_test.fasta \
--hyparam p_RES1_uniDist_best1 \
--tag $i \
--savemodel Y \
--outdir ../results/topt_models/TL1_RES1_ogt_onehot_SameAnces

done

```

#### Train from scratch
```
cd ../../scripts/
for i in {1..10}
do
python pre_train_simple.py \
--trainfile ../transfer_learning/Topt/data/cleaned_enzyme_topts_v1_train.fasta \
--modelname RES1 \
--hyparam p_RES1_uniDist_best1 \
--patience 200 \
--tag $i \
--outdir ../results/topt_models/Topt_Scratch
done

```


#### Test classical models
```
cd scripts
python evaluate_on_classical_models.py \
--infile ../data/AAC_0.csv \
--testfile ../data/cleaned_enzyme_topts_v1_test.fasta \
--outfile ../../results/topt_models/ClassicalModels/AAC_score.csv

python evaluate_on_classical_models.py \
--infile ../data/iFeatures.csv \
--testfile ../data/cleaned_enzyme_topts_v1_test.fasta \
--outfile ../../results/topt_models/ClassicalModels/iFeatures_score.csv

python evaluate_on_classical_models.py \
--infile ../data/unirep.csv \
--testfile ../data/cleaned_enzyme_topts_v1_test.fasta \
--outfile ../../results/topt_models/ClassicalModels/unirep_score.csv

```

