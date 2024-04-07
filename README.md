# MST-OATD
Code for "Multi-Scale Detection of Anomalous Spatio-Temporal Trajectories in Evolving Trajectory Datasets"
### Preprocessing
- Step1: Download data (<tt>train.csv.zip</tt>) from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data.
- Step2: Put the data file in <tt>../datasets/porto/</tt>, and unzip it as <tt>porto.csv</tt>.
- Step3: Run preprocessing by
```
mkdir -p data/porto
cd preprocess
python preprocess_porto.py
cd ..
```
### Generating ground truth
```
mkdir logs models
python generate_outliers.py --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset porto
```
distance is used to control the moving distance of outliers, fraction is the fraction of continuous outlier
### Training and testing
Example on the Porto dataset:
```
python train.py --task train --dataset porto
python train.py --task test --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset porto
```
### Training on evolving datasets
```
python train_labels.py
python train_update.py --update_mode pretrain --dataset porto --train_num 80000
```
update_mode contains three modes: pretrain, temporal, rank
