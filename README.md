# BE-521-Project

[Shared Folder](https://drive.google.com/drive/folders/0ANA1SbPXVwdfUk9PVA)

## Authors: Group MONITR

| Name        | Email                   |
| ----------- | ----------------------- |
| Kehan Chen  | chenkh@seas.upenn.edu   |
| Yining Guo  | gyn@seas.upenn.edu      |
| Yuetong Hao | yuetongh@seas.upenn.edu |

## Files

* `models/`: Stored models and parameters
  * `idx_S{subject_id}.npy`: selected feature indices for Subject {Subject_id}
  * `lgbr_f{finger_idx}_S{Subject_id}.txt`: saved LightGBM regressor models for finger {finger_idx} of Subject {Subject_id}
  * `XGB{Subject_id}.json`: saved XGB regressor models for Subject {Subject_id}
  * `NN_S{Subject_id}.pth`: saved MLP models for Subject {Subject_id}
  * `train_mean_S{Subject_id}.npy`: saved mean of training data
  * `train_std_S{Subject_id}.npy`: saved std of training data
* `utils/`: Includes all utils functions and classes for MLP:
  * `utils.py`: includes functions to construct features and R_matrix
  * `NN_model.py`: includes dataset class and the model class for the MLP `FingerRegressor`

* `train.ipynb`: training code of XGB Regressor and LightGBM Regressor
* `NN_Reg.ipynb`: training code of the MLP `FingerRegressor`
* `test.ipynb`: code to predict leaderboard data
* `truetest_prediction.ipynb`: code to predict hidden test data on colab
* `Algorithm_MONITR.zip`: zipped `utils/` and `models/`

## Workflow
![Flowchart](https://user-images.githubusercontent.com/65293070/235011914-b0c24c1a-4895-49b3-86cb-5c2672335a42.jpg)

