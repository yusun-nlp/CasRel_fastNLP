# CasRel_fastNLP
This repo is [fastNLP](https://fastnlp.readthedocs.io/zh/latest/) reimplementation of the paper:  [**"A Novel Cascade Binary Tagging Framework for Relational Triple Extraction"**](https://www.aclweb.org/anthology/2020.acl-main.136.pdf), which was published in ACL2020. The [original code](https://github.com/weizhepei/CasRel) was written in keras.

## Requirements

- Python 3.8
- Pytorch 1.7
- fastNLP 0.6.0
- keras-bert 0.86.0
- numpy 1.19.1 
- transformers 4.0.0

Other dependent packages described in [fastNLP Docs](https://fastnlp.readthedocs.io/zh/latest/user/installation.html).

## Datasets

- [NYT](https://github.com/weizhepei/CasRel/tree/master/data/NYT)
- [WebNLG](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG)

## Usage

2. **Build dataset in the form of triples** (DONE)
   Take the NYT dataset for example:
   
   a) Switch to the corresponding directory and download the dataset
   
   ```
   cd CasRel/data/NYT/raw_NYT
   ```
   
   b) Follow the instructions at the same directory, and just run
   
   ```
   python generate.py
   ```
   
   c) Finally, build dataset in the form of triples
   
   ```
   cd CasRel/data/NYT
   python build_data.py
   ```
   
   This will convert the raw numerical dataset into a proper format for our model and generate `train.json`, `test.json` and `dev.json`. Then split the test dataset by type and num for in-depth analysis on different scenarios of overlapping triples. (Run `python generate.py` under corresponding folders)
   
   - NYT:
     - Train: 56195, dev: 4999, test: 5000
     - normal : EPO : SEO = 3266 : 978 : 1297  
   - WebNLG:
     - Train: 5019, dev: 703, test: 500
     - normal : EPO : SEO = 182 : 16 : 318  
   
2. **Specify the experimental settings** (DONE)

   By default, we use the following settings in train.py:

   ```
   {
       "model_name": "CasRel",
       "dataset": "NYT",
       "bert_model_name": "bert-base-cased",
       "lr": 1e-5,
       "nulti-gpu": False,
       "batch_size": 6,
       "max_epoch": 100,
       "test_epoch": 5,
       "max_len": 100,
       "period": 50,
   }
   ```

3. **Train and select the model** 
   - [ ] Now trained in cpu, plan to move to gpu later.
   - [ ] Now define my own train model, plan to use Trainer class in fastNLP.

4. **Evaluate on the test set**



## Results



## References

[1] https://github.com/weizhepei/CasRel

[2] https://github.com/longlongman/CasRel-pytorch-reimplement

