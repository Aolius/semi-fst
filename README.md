
# Semi-Supervised Formality Style Transfer with Consistency Training (ACL2022)

## Dependencies

- python==3.8
- pytorch==1.6.0
- transformers==3.1.0
- nlpaug
- nltk
- kenlm
- pyskiplist
- statistics
- fitlog



## Dataset
### [GYAFC](https://github.com/raosudha89/GYAFC-corpus)
Please follow the guidance [here](https://github.com/raosudha89/GYAFC-corpus) to gain access to the Yahoo Answers L6 corpus and the GYAFC corpus.
Once you have gained access to the L6 corpus, please forward the acknowledgment to (zeitmond@gmail.com), and we will provide you the access to our unlabeled corpus collected from Yahoo Answers L6 corpus.

## Preparation
### 1: Pre-train style classifiers
```
python classifier/textcnn_t5.py -dataset em
```
### 2: Pre-train language models for the lm filter
After installing [kenlm](https://github.com/kpu/kenlm), use the following command to train a language model on the formal corpus of training data.
```
bin/lmplz -o 4 <em_formal.txt >em_formal.arpa
```
**Note**: We also provide pretrained checkpoints in `./checkpoints/`.


## Training
```
sh run_sup.sh 
sh run_semi.sh
```
Check `T5_Large_filter.py` for detailed descriptions of the hyperparameters.

**Important notes:** When using the `spell` data perturbation from nlpaug, there will be unexpected randomness in the augmented sentences, making the training results irreproducible.
To avoid this, check `./aug_fix/README.md`.

## Model Outputs
The outputs of our best systems are provided in `./model_outputs`.


