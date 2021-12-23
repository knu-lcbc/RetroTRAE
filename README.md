## RetroTRAE: RETROSYNTHETIC TRANSLATION OF ATOMIC ENVIRONMENTS WITH TRANSFORMER
We present a new single-step retrosynthesis prediction method, viz. RetroTRAE, using fragment-based tokenization and the Transformer architecture. RetroTRAE predicts reactant candidates by learning the changes of atom environments (AEs) associated with the chemical reaction. AEs are the ideal stand-alone chemically meaningful building blocks providing a high-resolution molecular representation. Describing a molecule with a set of AEs establishes a clear relationship between translated product-reactant pairs due to the conservation of atoms in the reactions. Our approach introduces a novel scheme for fragmental and topological descriptors to be used as natural inputs for retrosynthetic prediction tasks.

<hr style="background: transparent; border: 0.2px dashed;"/>

### Datasets
#### Training

We used a curated subset of Lowe’s grants dataset (USPTO-Full). [Jin et al.](https://github.com/wengong-jin/nips17-rexgen) further refined the USPTO-Full set by removing duplicates and erroneous reactions. This curated dataset contains 480K reactions.
Preprocessing steps to remove reagents from reactants are described by [Liu et al.](https://github.com/pandegroup/reaction_prediction_seq2seq) and [Schwaller et al.](https://github.com/ManzoorElahi/organic-chemistry-reaction-prediction-using-NMT).

We used [Zheng's version](https://github.com/sysu-yanglab/Self-Corrected-Retrosynthetic-Reaction-Predictor/tree/master/data) (removal of agents and canonicalization of Jin's dataset) of USPTO and carefully curated the product-reactant pairs. 
We generated two distinct curated datasets consist of unimolecular and bimolecular reactions, with sizes 100K and 314K respectively.
Since retrosynthesis implies an abstract backward direction, we named our datasets unimolecular and bimolecular reactions.
There was no reaction class information available in this dataset and we have not used any atom-to-atom mapping algorithm.

#### Post-processing

Additionally, [PubChem compound database (_CID-SMILES.gz_)](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/), 111 million compounds, is used to recover molecules from a list of AEs.

<hr style="background: transparent; border: 0.5px dashed;"/>

### Code usage

#### Requirements
The source code is tested on Linux operating systems. We recommend creating a new conda environment. Users should install required packages described in _environments.yml_ prior to direct use.

   ```shell
   conda env create --name RetroTRAE_env --file=environments.yml
   ```

#### Configurations:

1. Users can set various hyperparameters in `src/parameters.py` file.

2. The following command `src/tokenizer_with_split.py` applies tokenization scheme and also splits the data.

   ```shell
   python src/tokenizer_with_split.py --model_type='bi'
   ```
   - `--model_type`: By default, it runs for bimolecular reaction dataset.

   In default setting, the structure of whole data directory should be like below.

   - `data`
     - `sp`
       - `src_sp.model`, `src_sp.vocab`, `tar_sp.model`, `tar_sp.vocab`
     - `src`
       - `train.txt`, `valid.txt`, `test.txt`
     - `trg`
       - `train.txt`, `valid.txt`, `test.txt`
     - `raw_data.src`
     - `raw_data.trg`

<hr style="background: transparent; border: 0.5px dashed;"/>
   
#### Prediction & Demo:

 First, download [checkpoints files](https://drive.google.com/drive/folders/1lntDBIEt4Yz9Iv1YBez3pke458URhMhZ?usp=sharing) and extract the archive file.
  
 Run below commands to conduct an inference with the trained model.

   ```shell
   python src/predict.py  --smiles --decode --uni_checkpoint_name --bi_checkpoint_name
   ```
   - `--smiles`: The molecule we wish to synthetize.
   - `--decode`: Decoding algorithm (either `'greedy'` or `'beam'`), (by default: `greedy`)
   - `--uni_checkpoint_name`: Checkpoint file name for unimolecular rxn model. (default: `uni_checkpoint.pth`)
   - `--bi_checkpoint_name`: Checkpoint file name for bimolecular rxn model. (default: `bi_checkpoint.pth`)

#### Training:
 Run below command to train a transformer model for retrosynthetic prediction.

   ```shell
   python src/train.py --model_type --resume=False --custom_validation=False --checkpoint_name=CHECKPOINT_NAME
   ```
   - `--model_type`: Select either `'uni'` for unimolecular reactions or `'bi'` for bimolecular reactions
   - `--resume`: Resume training for a given checkpoint. (default: `False`)
   - `--custom_validation`: Evaluates the model accuracy based on the custom metrics. (default: `False`)
   - `--checkpoint_name`: This specify the checkpoint file name. (default: `None`)
   

   
<hr style="background: transparent; border: 0.5px dashed;"/>

   
#### Accuracy comparison in the paper

**Model performance comparison without additional reaction classes.**

The below results are based on either filtered MIT-full or MIT-fully atom mapped reaction datasets.
    
| Model       | top-1 accuracy (%)                         |
| -------------------- | ------------------------------------------------------------ |
| `Non-Transformer`       |                    |
| Coley et al., similarity-based, 2017      | 32.8                   |
| Segler et al.,--rep. by Lin, Neuralsym, 2020 |  47.8                  |
| Dai et al., Graph Logic Network, 2019 | 39.3                 |
| Liu et al.,--rep. by Lin, LSTM-based, 2020 | 46.9                   |
| Genheden et al., AiZynthfinder, ANN + MCTS, 2020 | 43-72                   |
| `Transformer-based`       |                    |
| Zheng et al., SCROP, 2020 | 41.5              |
| Wang et al., RetroPrime, 2021      | 44.1                   |
| Tetko et al., Augmented Transformer, 2020     |  46.2                  |
| Lin et al., AutoSynRoute, Transformer + MCTS, 2020     |      54.1              |
| RetroTRAE     |     58.3               |
| RetroTRAE (with SM and DM)      | 61.6                   |
