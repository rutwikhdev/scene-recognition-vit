## Install Dependencies

1. Download pytorch
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
2. Clone repo and nstall rest of the requirements
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
--model-name vit_base \  # ["vit_base", "swin"]
--batch-size 128
```
Running log file is saved in logs/ directory and tensorboard data is saved in runs/ which can be visualized as ```tensorboard --logdir=runs```
