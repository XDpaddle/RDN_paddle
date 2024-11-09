# RDN_paddle

# RDN

This repository is implementation of the ["Residual Dense Network for Image Super-Resolution"](https://arxiv.org/abs/1802.08797).



## Requirements

- paddlepaddle 2.4.0
- paddleseg    2.8.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Train

The DIV2K, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type  | Link                                                         |
| ------- | ----- | ----- | ------------------------------------------------------------ |
| DIV2K   | 2     | Train | [Download](https://www.dropbox.com/s/41sn4eie37hp6rh/DIV2K_x2.h5?dl=0) |
| DIV2K   | 3     | Train | [Download](https://www.dropbox.com/s/4piy2lvhrjb2e54/DIV2K_x3.h5?dl=0) |
| DIV2K   | 4     | Train | [Download](https://www.dropbox.com/s/ie4a6t7f9n5lgco/DIV2K_x4.h5?dl=0) |
| Set5    | 2     | Eval  | [Download](https://www.dropbox.com/s/b7v5vis8duh9vwd/Set5_x2.h5?dl=0) |
| Set5    | 3     | Eval  | [Download](https://www.dropbox.com/s/768b07ncpdfmgs6/Set5_x3.h5?dl=0) |
| Set5    | 4     | Eval  | [Download](https://www.dropbox.com/s/rtu89xyatbb71qv/Set5_x4.h5?dl=0) |

Otherwise, you can use `prepare.py` to create custom dataset.

### prepare dataset

```bash
python prepare.py --images-dir "/root/autodl-tmp/paddle-FSRCNN/SR/WDSR/DIV2K/DIV2K_train_HR" \
                  --output-path "/root/autodl-tmp/paddle-FSRCNN/SR/DIV2K_X3.h5" \
                  --scale 3
```
### train
```bash
python train.py --train-file "/root/autodl-tmp/paddle-FSRCNN/SR/DATA/DIV2K_X3.h5" \
                --eval-file "/root/autodl-tmp/paddle-FSRCNN/SR/RDN/BLAH_BLAH/Set5_x3 .h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 3 \
                --num-features 64 \
                --growth-rate 64 \
                --num-blocks 16 \
                --num-layers 8 \
                --lr 1e-4 \
                --batch-size 16 \
                --patch-size 32 \
                --num-epochs 800 \
                --num-workers 0 \
                --seed 123                
```



## Test

Pre-trained weights can be downloaded from the links :链接：https://pan.baidu.com/s/17aVKmAG_k_Ag1Uiq1OCwjA?pwd=7nds 
提取码：7nds

```bash
python test.py --weights-file "/root/autodl-tmp/paddle-FSRCNN/SR/RDN/BLAH_BLAH/outputs/x2/best.pdiparams" \
               --image-file "data/119082.png" \
               --scale 2 \
               --num-features 64 \
               --growth-rate 64 \
               --num-blocks 16 \
               --num-layers 8

```

## Results

PSNR was calculated on the Y channel.



