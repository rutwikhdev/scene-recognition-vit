## Install Dependencies

1. Download pytorch
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
2. Clone repo and install rest of the requirements
```
git clone https://github.com/rutwikhdev/scene-recognition-vit && cd scene-recognition-vit
pip install -r requirements
```
3. Download custom scene-recognition dataset and place it in ```/data/{dataset_name}```

## Run Experiments
```
python train.py \
--data-dir ./data/Places2_simp \
--epochs 10 \
--lr 3e-4 \
--scheduler cosinealr \  # ["cosine", "cosinealr", "onecyclelr"]
--model-name vit_base \  # ["vit_base", "swin", "swinv2", "dino"]
--batch-size 128
```
Running log file is saved in logs/ directory and tensorboard data is saved in runs/ which can be visualized as ```tensorboard --logdir=runs```

## Best Results
> [!Note]
> All the results were generated with a batch-size of 128 and the accuracies mentioned are for validation set(randomly split in dataset.py)

| Model   | Learning Rate | LR Scheduler    | Top-1 Accuracy | Top-5 Accuracy | Epochs | Log File                |
|---------|:-------------:|----------------:|:--------------:|:--------------:|:------:|------------------------|
| vit_base | 3e-4          | None          | 70.1%          | 93.4%          |  10    | [Log](https://github.com/rutwikhdev/scene-recognition-vit/blob/main/logs/log_rh01555_20250511_161503/log_rh01555_20250511_161503.txt)    |
| swin | 3e-4         | None | 66.3%          | 92.6%          | 10    | [Log](https://github.com/rutwikhdev/scene-recognition-vit/blob/main/logs/log_rh01555_20250511_161504/log_rh01555_20250511_161504.txt)    |
| swinv2 | 3e-4         | None     | 66.1%          | 92.5%          |  10   | [Log](https://github.com/rutwikhdev/scene-recognition-vit/blob/main/logs/log_rh01555_20250511_164752/log_rh01555_20250511_164752.txt)    |
| vit_base | 4e-4         | CosineAnnealingLR     | 70.0%          | 93.4%          |  10   | [Log](https://github.com/rutwikhdev/scene-recognition-vit/blob/main/logs/log_rh01555_20250511_173323/log_rh01555_20250511_173323.txt)    |
| dinov2 | 3e-4         | CosineAnnealingLR | 80.9%          | 97.6%          |  10   | [Log](https://github.com/rutwikhdev/scene-recognition-vit/blob/main/logs/log_rh01555_20250511_170355/log_rh01555_20250511_170355.txt)    |


## Visualising Pre-trained models
We have included 3 notebooks,
1. 01_data_analysis.ipynb - Basic data visualizations to understand what we are working with
2. 02_validation_analysis.ipynb - Evaluation on validation set and visualizations.
3. 03_testset.ipynb - Can run same evaluations as validation on [test_set](https://drive.google.com/drive/folders/1vsIbblGiXFPAWVWG_dmkOqhMtezGlczj?usp=sharing) or predict classes for individual images. This notebook also visualizes attention maps

Alternatively after installing all the dependencies above you can run,
```
python3 download_models.py
streamlit run stream.py
```
This will provide an interactive UI

![Screenshot From 2025-05-12 13-19-22](https://github.com/user-attachments/assets/53b1d2c0-11af-4282-b7b0-093aed60072f)
![Screenshot From 2025-05-12 13-19-38](https://github.com/user-attachments/assets/50b13286-1d1e-4787-939b-aad5f46f51fa)


