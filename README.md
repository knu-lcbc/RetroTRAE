[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/) 
[![DOI](https://zenodo.org/badge/394543593.svg)](https://zenodo.org/badge/latestdoi/394543593)


## RetroTRAE: Retrosynthetic Translation of Atomic Environments with Transformer
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

***UPDATED**:
Now you can generate the PubChem Database for retrieval with the following command after downloading/extracting *CID-SMILES*. 
[PubChem compound database (_CID-SMILES.gz_)](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz), 111 million compounds, is used to recover molecules from a list of AEs.
```shell
    python pubchem_gen.py --cid-smiles-path /path/to/CID-SMILES
```
For more details `python pubchem_gen.py --help`

<hr style="background: transparent; border: 0.5px dashed;"/>

### Code usage

#### Requirements
The source code is tested on Linux operating systems. After cloning the repository, we recommend creating a new conda environment. Users should install required packages described in _environments.yml_ prior to direct use.

   ```shell
   conda create --name RetroTRAE_env python=3.8 -y
   conda activate RetroTRAE_env
   conda install pytorch cudatoolkit=11.3 -c pytorch -y
   conda install -c conda-forge rdkit -y
   pip install sentencepiece
   
   ```
   or

   ```shell
   conda env create --name RetroTRAE_env --file=environment.yml
   ```
   
#### Prediction & Demo:

First, [checkpoints files](https://drive.google.com/file/d/1ZIIFwLzFuJEBTl1Zwq3qBEjOvejN4O5H/view?usp=sharing) should be downloaded and extracted.

Run below commands to conduct an inference with the trained model.
 
***UPDATED**: This new code will be default in future. 
  ```shell
   python predict.py --smiles='COc1cc2c(c(Cl)c1OC)CCN(C)CC2c1ccccc1' 
   ```
   ```shell
--smiles SMILES       An input sequence (default: None)                                                                                                          
  --decode {greedy,beam}                                                                                                                                           
                        Decoding method for RetroTRAE (default: greedy)                                                                                            
  --beam_size BEAM_SIZE                                                                                                                                            
                        Beam size (a number of candidates for RetroTRAE) (default: 3)                                                                              
  --conversion {ml,db}  How to convert AEs to SMILES? 'ml': Machine Learning model 'db': Retrieve from PubChem database (default: ml)                              
  --database_dir DATABASE_DIR                                                                                                                                      
                        Database for retrieval of the predicted molecules (default: ./data/PubChem_AEs)                                                            
  --topk TOPK           A number of candidates for the AEs to SMIES conversion (default: 1)                                                                        
  --uni_checkpoint_name UNI_CHECKPOINT_NAME                                                                                                                        
                        Checkpoint file name (default: uni_checkpoint.pth)                                                                                         
  --bi_checkpoint_name BI_CHECKPOINT_NAME                                                                                                                          
                        Checkpoint file name (default: bi_checkpoint.pth)                                                                                          
  --log_file LOG_FILE   A file name for saving outputs (default: None)
   ```
   
 ***Note**: The old code is still there. 


   ```shell
   python src/predict.py  --smiles
   ```
   - `--smiles`: The molecule we wish to synthetize.
   - `--decode`: Decoding algorithm (either `'greedy'` or `'beam'`), (by default: `greedy`)
   - `--uni_checkpoint_name`: Checkpoint file name for unimolecular rxn model. (default: `uni_checkpoint.pth`)
   - `--bi_checkpoint_name`: Checkpoint file name for bimolecular rxn model. (default: `bi_checkpoint.pth`)
   - `--database_dir`: Path containing DB files.

Example prediction and sample output;

Results are saving to InChIKey coded filename.

   ```shell
   >> python src/predict.py --smiles='COc1cc2c(c(Cl)c1OC)CCN(C)CC2c1ccccc1' --database_dir DB_Path

   unimolecular model is building...
   Loading checkpoint...

   bimolecular model is building...
   Loading checkpoint...

   greedy decoding searching method is selected.
   Preprocessing input SMILES: COc1cc2c(c(Cl)c1OC)CCN(C)CC2c1ccccc1
   Corresponding AEs: [c;R;D3](-[CH;R;D3])(:[c;R;D3]):[cH;R;D2] ... [c;R;D3](-[Cl;!R;D1])(:[c;R;D3]):[c;R;D3] [CH;R;D3]


   Predictions are made in AEs form.
   Saving the results here: results_VDCYGTBDVYWJFQ-UHFFFAOYSA-N.json

   Done!
   ```

#### Training:

#### Configurations:

1. Users can set various hyperparameters in `src/parameters.py` file.

2. The following command `src/tokenizer_with_split.py` applies tokenization scheme and also splits the data.

   ```shell
   python src/tokenizer_with_split.py --model_type='bi'
   ```
   - `--model_type`: By default, it runs for bimolecular reaction dataset. 

The structure of whole data directory should be prefixed by `model_type`.

   - `data`
     - `sp`
       - `src_sp.model`, `src_sp.vocab`, `tar_sp.model`, `tar_sp.vocab`
     - `src`
       - `train.txt`, `valid.txt`, `test.txt`
     - `trg`
       - `train.txt`, `valid.txt`, `test.txt`
     - `raw_data.src`
     - `raw_data.trg`

Below command can be simply used to train the model for retrosynthetic prediction.

   ```shell
   python src/train.py --model_type='bi'
   ```
   - `--model_type`: `'uni'` or `'bi'`. (default: `bi`)
   - `--custom_validation`: Evaluates the model accuracy based on the custom metrics. (default: `True`)
   - `--resume`: Resume training for a given checkpoint. (default: `False`)
   - `--start_epoch`: Epoch numbers for resumed training (default: `0`)
   - `--checkpoint_name`: Checkpoint file name. (default: `None`)
   
   
<hr style="background: transparent; border: 0.5px dashed;"/>

   
### Results

Model performance comparison without additional reaction classes based on either filtered MIT-full or Jin's USPTO.
    

| Model                                               | top-1  | top-3  | top-5  | top-10  |
| --------------------------------------------------- | ------ | ------ | ------ | ------- |
| `Non-Transformer`                                   |        |        |        |         |
| Coley et al., similarity-based, 2017                | 32.8   |        |        |         |
| Segler et al.,--rep. by Lin, Neuralsym, 2020        | 47.8   | 67.6   | 74.1   | 80.2    |
| Dai et al., Graph Logic Network, 2019               | 39.3   |        |        |         |
| Liu et al.,--rep. by Lin, LSTM-based, 2020          | 46.9   | 61.6   | 66.3   | 70.8    |
| Genheden et al., AiZynthfinder, ANN + MCTS, 2020    | 43-72  |        |        |         |
| `Transformer-based`                                 |        |        |        |         |
| Zheng et al., SCROP, 2020                           | 41.5   |        |        |         |
| Wang et al., RetroPrime, 2021                       | 44.1   |        |        |         |
| Tetko et al., Augmented Transformer, 2020           | 46.2   |        |        |         |
| Lin et al., AutoSynRoute, Transformer + MCTS, 2020  | 54.1   | 71.8   | 76.9   | 81.8    |
| RetroTRAE                                           | 58.3   | 66.1   | 69.4   | 73.1    |
| RetroTRAE (with SM and DM)                          | 61.6   |        |        |         |

<hr style="background: transparent; border: 0.5px dashed;"/>

### License

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
