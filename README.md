# KFSM: Knowledge Flow Selection Mechanism
Codes for the paper: "Modeling and utilizing the dynamic Knowledge Flow in Multi-turn Conversations by Using Previous Knowledge Selections"

## Requirements

- pytorch >= 1.1
- python >= 3.6
- tqdm
- cotk==0.0.1
- numpy
- nltk

## Datasets

### Download
Download the [Wizard of Wikipedia](https://drive.google.com/drive/folders/1eowwYSfJKaDtYgKHZVqh8alNmqP3jv9A?usp=sharing) dataset (downloaded using [Parlai](https://github.com/facebookresearch/ParlAI), put the files in the folder `./Wizard-of-Wikipedia`, or download the [Holl-E](https://drive.google.com/drive/folders/1xQBRDs5q_2xLOdOpbq7UeAmUM0Ht370A?usp=sharing) dataset and put the files in the folder `./Holl-E`.

### Prepare
For Wizard of Wikipedia (WoW):

```bash
python prepare_wow_data.py
```

For Holl-E:

```bash
python prepare_holl_data.py
```

Besides, download the pretrained [wordvector](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip), unzip the files in `./` and rename the 300-d embedding file as `glove.txt`.

## Training

For Holl-E:

```bash
python run.py \
    --mode train \
    --dataset HollE \
    --datapath ./Holl-E/prepared_data \
    --wvpath ./ \
    --cuda 0 \
    --droprate 0.5 \
    --hist_len 4 \
    --hist_weights 0.25 0.25 0.25 0.25 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

For Wizard of Wikipedia:

```bash
python run.py \
    --mode train \
    --dataset WizardOfWiki \
    --datapath ./Wizard-of-Wikipedia/prepared_data \
    --wvpath ./ \
    --cuda 0 \
    --droprate 0.5 \
    --hist_len 4 \
    --hist_weights 0.25 0.25 0.25 0.25 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

## Evaluation

For Holl-E:

```bash
python run.py \
    --mode test \
    --dataset Holl-E \
    --cuda 0 \
    --restore best \
    --hist_len 4 \
    --hist_weights 0.25 0.25 0.25 0.25 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

For Wizard of Wikipedia:

```bash
python run.py \
    --mode test \
    --dataset WizardOfWiki \
    --cuda 0 \
    --restore best \
    --hist_len 4 \
    --hist_weights 0.25 0.25 0.25 0.25 \
    --out_dir ./output \
    --model_dir ./model \
    --cache
```

## Addition

More descriptions will be released in a few days...

