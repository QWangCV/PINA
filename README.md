# PINA

This is the official implementation of our ECCV 2024 paper:  
_Non-Exemplar Domain Incremental Learning via Cross-Domain Concept Integration_


## Environment

```bash
conda create -n pina python=3.8
conda activate pina
pip install -r requirements.txt
```

## Datasets

### DomainNet
Please refer to [DomainNet Project](http://ai.bu.edu/M3SDA/) to download the dataset or run:
```bash
cd datasets
bash download_domainnet.sh
```
Then unzip the downloaded files, and confirm the file directory as shown below:
```
DomainNet
├── clipart
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── clipart_test.txt
├── clipart_train.txt
├── infograph
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── infograph_test.txt
├── infograph_train.txt
├── painting
│   ├── aircraft_carrier
│   ├── airplane
... ...
```
### CDDB
Please refer to [CDDB Project](https://github.com/Coral79/CDDB) and download the dataset from [CDDB Dataset](https://drive.google.com/file/d/1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg/view?usp=sharing).

Then unzip the downloaded files, and confirm the file directory as shown below:
```
CDDB
├── biggan
│   ├── train
│   └── val
├── gaugan
│   ├── train
│   └── val
├── san
│   ├── train
│   └── val
├── whichfaceisreal
│   ├── train
│   └── val
├── wild
│   ├── train
│   └── val
... ...
```


### CORe50
Please refer to [CORe50 Project](https://vlomonaco.github.io/core50/index.html#dataset) and download the file shown below:
```
CORe50
├── core50_imgs.npz
├── labels.pkl
├── LUP.pkl
└── paths.pkl
```




## Training and Inference
Please confirm the path of your datasets in the config files.
### DomainNet
```
python main.py --config configs/domainnet_pina_vit.yaml --device 0
python main.py --config configs/domainnet_pina_clip.yaml --device 0
```

### CDDB
```
python main.py --config configs/cddb_pina_vit.yaml --device 0
python main.py --config configs/cddb_pina_clip.yaml --device 0
```

### CORe50
```
python main.py --config configs/core50_pina_vit.yaml --device 0
python main.py --config configs/core50_pina_clip.yaml --device 0
```

## Acknowledgement
We thank [PyCIL](https://github.com/G-U-N/PyCIL) and [S-Prompts](https://github.com/iamwangyabin/S-Prompts) for their wonderful framework and codes!  
We also thank [CLIP](https://github.com/openai/CLIP) and [CoOp](https://github.com/KaiyangZhou/CoOp) for their helpful components.


## Citation
If any part of our paper and code is helpful to your research, please consider citing the following bib entry:
```
@inproceedings{wang2024non,
  title={Non-exemplar domain incremental learning via cross-domain concept integration},
  author={Wang, Qiang and He, Yuhang and Dong, Songlin and Gao, Xinyuan and Wang, Shaokun and Gong, Yihong},
  booktitle={European Conference on Computer Vision},
  pages={144--162},
  year={2024},
  organization={Springer}
}
```
