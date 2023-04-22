# BE-521-Project

[Shared Folder](https://drive.google.com/drive/folders/0ANA1SbPXVwdfUk9PVA)

## Authors

| Name        | Email                   |
| ----------- | ----------------------- |
| Kehan Chen  | chenkh@seas.upenn.edu   |
| Yining Guo  | yuetongh@seas.upenn.edu |
| Yuetong Hao | gyn@seas.upenn.edu      |

## Files

* `models/`: Stored models and parameters
  * `idx_S{subject_id}.npy`: selected feature indices for Subject {Subject_id}
  * `lgbr_f{finger_idx}_S{Subject_id}.txt`: saved LightGBM regressor models for finger {finger_idx} of Subject {Subject_id}
  * `XGB{Subject_id}.json`: saved XGB regressor models for Subject {Subject_id}
  * `NN_S{Subject_id}.pth`: saved MLP models for Subject {Subject_id}
  * `train_mean_S{Subject_id}.npy`: saved mean of training data
  * `train_std_S{Subject_id}.npy`: saved std of training data
* `train.ipynb`: training code of XGB Regressor and LightGBM Regressor
* `test.ipynb`: code to predict leaderboard data
* `truetest_prediction.ipynb`: code to predict hidden test data on colab.

## Usage

## Workflow
