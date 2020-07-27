# AutoEncoder with pytorch
This is a pytorch implementation of AutoEncoder. 

## Requirements
```bash
$ pip install -r requirements.txt
```

## Usage
### Configs
Create a configuration file based on configs/default.yaml.
```bash
### dataset
data_root: {path to dataset}
img_size: 32
n_channels: 3
color_mean: [0.4914, 0.4822, 0.4465]
color_std: [0.2023, 0.1994, 0.2010]

### train parameters
lr: 0.0001
decay: 1e-4
n_gpus: 1
batch_size: 64
n_epochs: 50

# save_ckpt_interval should not be 0.
save_ckpt_interval: 50

# output dir (logs, results)
log_dir: ./logs/

# checkpoint path or None
resume: 
# e.g) resume: ./logs/2020-07-26T00:19:34.918002/ckpt/best_acc_ckpt.pth
```

### Prepare Dataset
You need to prepare a directory with the following structure:
```bash
datasets/
├── images
│   ├── hoge.png
│   ├── fuga.png
│   ├── foo.png
│   └── bar.png
├── train.csv
└── test.csv
```

The content of the csv file should have the following structure.
```bash
filename,     label
airplane1.png,0
car1.png,1
cat1.png,3
deer1.png,4
```

An example of a dataset can be found in the dataset folder.

### Train
```bash
$ python main.py --config ./configs/default.yaml
```

### Inference
```bash
$ python main.py --config ./configs/default.yaml --inference
```

### Tensorboard
```bash
tensorboard --logdir={log_dir} --port={your port}
```
![tensorboard](docs/images/tensorboard.png)

## Output
You will see the following output in the log directory specified in the Config file.
```bash
# Train
logs/
└── 2020-07-26T14:21:39.251571
    ├── checkpoint
    │   ├── best_acc_ckpt.pth
    │   ├── epoch0000_ckpt.pth
    │   └── epoch0001_ckpt.pth
    ├── metrics
    │   ├── train_metrics.csv
    │   └── test_metrics.csv 
    ├── tensorboard
    │   └── events.out.tfevents.1595773266.c47f841682de
    └── logfile.log

# Inference
inference_logs/
└── 2020-07-26T14:21:06.197407
    ├── images (only if inference mode)
    │   ├── hoge.png
    │   └── huga.png
    ├── tensorboard
    │   └── events.out.tfevents.1595773266.c47f841682de
    └── logfile.log
```

The contents of train_metrics.csv and test_metrics.csv look like as follows:
```bash
epoch,train loss,
0,0.024899629971981047
1,0.020001413972377778
