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
mkdir logs models probs
```
The pre-processed Chengdu sample data can be downloaded from https://www.dropbox.com/scl/fo/fxunogur65jbqhs69d2m6/h?rlkey=z8rzg0lxrtsx77bkeaunojq2i&dl=0.

### Generating ground truth

```
python generate_outliers.py --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset <dataset_name>
```
distance is used to control the moving distance of outliers, fraction is the fraction of continuous outlier, obeserved_ratio is the ratio of the obeserved part of a trajectory.
### Training and testing
Example on the Porto dataset:
```
python train.py --task train --dataset <dataset_name>
python train.py --task test --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset <dataset_name>
```
### Training on evolving datasets
```
python train_labels.py
python train_update.py --update_mode pretrain --dataset <dataset_name> --train_num <train_num>
```
update_mode contains three modes: pretrain, temporal, rank, <train_num> is the number of trajectories used for evolving training.
