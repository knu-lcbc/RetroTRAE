## RETROTRAE: RETROSYNTHETIC TRANSLATION OFATOMIC ENVIRONMENTS WITH TRANSFORMER
Abstract.We present the new retrosynthesis prediction methodRetroTRAE using fragment-based tokenization combined with theTransformer  architecture.   RetroTRAE  represents  chemical  reac-tions by using the changes of fragment sets of molecules using theatomic  environment  fragmentation  scheme.   Atom  environmentsstand as an ideal, chemically meaningful building blocks togetherproducing a high resolution molecular representation.  Describinga  molecule  with  a  set  of  atom  environments  establishes  a  clearrelationship between translated product-reactant pairs due to con-servation of atoms in reactions.  Our model achieved a top-1 ac-curacy of 67.1% within the bioactively similar range for USPTOtest dataset, outperforming the other state of the art, translationmethods.  We investigated the effect of different encoding scenarioson predicting the reactant candidates.  We also critically assessedthe retrieval process that converts a set of fragments into a mole-cule with respect to coverage, degeneracy and resolution.  Our newtemplate-free model for retrosynthetic prediction provides fast andreliable  retrosynthetic  route  planning  for  substances  whose  frag-mentation patterns are revealed.

<br/>
<hr style="background: transparent; border: 0.5px dashed;"/>

### Requirements
* Python (version >= 3.6) 
* Numpy (version >= 1.16.4) 
* PyTorch (version >= 1.0.0) 
* RDKit (version >= 2019)

 Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

## Datasets
We used a subset of the filtered US patent reaction dataset, USPTO-Full, which is obtained with a text-mining approachi by Lowe.
This subset contains 480K atom-mapped reactions.
For training our models, atom-mapping information was not used. 
Also, there are no reaction class information is available in this dataset.
We generated two distinct curated datasets consist of unimolecular and bimolecular reactions, with sizes 100K and 314K respectively. 
Additionally, PubChem compound database (111M) is used to recover molecules from a list of AEs.

<hr style="background: transparent; border: 0.5px dashed;"/>


## Usage

### Configurations:

1. You can set various hyperparameters in `src/parameters.py` file.


2. Run `src/tokenizer_with_split.py`.

   ```shell
   python src/tokenizer_with_split.py
   ```

   Then there would be `SP_DIR` directory containing two sentencepiece models and two vocab files.

   Each model and vocab files are for source language and target language and it splits the dataset into training and validation set.

   In default setting, the structure of whole data directory should be like below.

   - `data`
     - `sp`
       - `src_sp.model`
       - `src_sp.vocab`
       - `tar_sp.model`
       - `tar_sp.vocab`
     - `src`
       - `train.txt`
       - `valid.txt`
       - `test.txt`
     - `trg`
       - `train.txt`
       - `valid.txt`
       - `test.txt`
     - `raw_data.src`
     - `raw_data.trg`

   <br/>

### Training:
 Run below command to train a transformer model for retrosynthetic prediction.

   ```shell
   python src/train.py --resume=False --custom_validation=False --checkpoint_name=CHECKPOINT_NAME
   ```
   - `--resume`: Resume training for a given checkpoint. (default: `False`)
   - `--custom_validation`: Evaluates the model accuracy based on the custom metrics. (default: `False`)
   - `--checkpoint_name`: This specify the checkpoint file name. (default: `None`)
   

   <br/>

### Prediction:
 Run below command to conduct an inference with the trained model.

   ```shell
   python src/predict.py  --input=INPUT_TEXT --decode=DECODING_METHOD --checkpoint_name=CHECKPOINT_NAME 
   ```
   - `--input`: This is an input sequence you want to translate.
   - `--decode`: This makes the decoding algorithm into either greedy method or beam search. Make this parameter 'greedy' or 'beam'.  (default: `greedy`)
   - `--checkpoint_name`: This specify the checkpoint file name. (default: `best_checkpoint.pth`)

   <br/>
   
<hr style="background: transparent; border: 0.5px dashed;"/>



    
## Accuracy comparison in the paper
**Model performance comparison without additional reaction classes.**

The below results are based on either filtered MIT-full or MIT-fully atom mapped reaction datasets.
    
| Model       | top-1 accuracy (%)                         |
| -------------------- | ------------------------------------------------------------ |
| `Non-Transformer`       |                    |
| Coley et al., Similarity, 2017       | 32.8                   |
| Segler et al., Neuralsym, 2017 | 35.8                  |
| Segler-Coley,--rep. by Lin, 2020 | 47.8                   |
| Dai et al., GLN, 2019 | 39.3                 |
| Liu et al.--rep. by Lin, 2020 | 46.9                   |
| `Transformer-based`       |                    |
| Zheng et al., SCROP, 2020 | 41.5              |
| Wang et al., RetroPrime, 2021      | 44.1                   |
| Tetko et al., AT, 2020      |  46.2                  |
| Lin et al., 2020      |      54.1              |
| RetroTRAE -- this work      |     53.4               |
| RetroTRAE + Bioactive -- this work      | 67.1                   |