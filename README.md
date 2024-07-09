# MST-OATD
Code for "Multi-Scale Detection of Anomalous Spatio-Temporal Trajectories in Evolving Trajectory Datasets"
### Preprocessing
- Step1: Download the Porto dataset (<tt>train.csv.zip</tt>) from [Porto](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data).
- Step2: Put the Porto data file in <tt>../datasets/porto/</tt>, and unzip it as <tt>porto.csv</tt>. Put the unzipped Chengdu data in <tt>../datasets/chengdu/</tt>.
- Step3: Run preprocessing by
```
mkdir -p data/<dataset_name>
cd preprocess
python preprocess_<dataset_name>.py
cd ..
mkdir logs models probs
```
 <dataset_name>:  <tt>porto</tt> or  <tt>cd</tt>

### Generating ground truth

```
python generate_outliers.py --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset <dataset_name>
```
distance is used to control the moving distance of outliers, fraction is the fraction of continuous outlier, obeserved_ratio is the ratio of the obeserved part of a trajectory.
### Training and testing
```
python train.py --task train --dataset <dataset_name>
python train.py --task test --distance 2 --fraction 0.2 --obeserved_ratio 1.0 --dataset <dataset_name>
```
### Training on evolving datasets
```
python train_labels.py --dataset <dataset_name>
python train_update.py --update_mode pretrain --dataset <dataset_name> --train_num <train_num>
```
update_mode contains three modes: <tt>pretrain</tt>, <tt>temporal</tt>, <tt>rank</tt>, <train_num> is the number of trajectories used for evolving training.
