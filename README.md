# uresnet_pytorch
PyTorch implementations of dense and sparse UResNet retrofitted to run on NERSC (or any system without GPU support).

## Running UResnet

To train,

```
python bin/uresnet.py train -chks 500 -wp weights/snapshot -io larcv_sparse -bs 1 --gpus 0 -nc 5 -rs 1 -ss 512 -dd 3 -uns 5 -uf 16 -dkeys data,label -mn uresnet_sparse -it 10 -ld log -if your_data.root
```

Main command-line parameters:
* `-mn` model name, can be `uresnet_dense` or `uresnet_sparse`
* `-io` I/O type, can be `larcv_sparse` or `larcv_dense`
* `-nc` number of classes
* `-chks` save checkpoint every N iterations
* `-wp` weights directory
* `-bs` batch size
* `--gpus` list gpus
* `-rs` report every N steps in stdout
* `-ss` spatial size of images
* `-dd` data dimension (2 or 3)
* `-uns` U-ResNet depth
* `-uf` U-ResNet initial number of filters
* `-dkeys` data keys in LArCV ROOT file
* `-it` number of iterations
* `-ld` log directory
* `-if` input file
* `-mp` weight files to load for inference
