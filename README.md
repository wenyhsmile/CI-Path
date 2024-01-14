<!-- #region -->
# <ins>P</ins>ath-based <ins>G</ins>raph Neural Network <ins>E</ins>xplanation for Heterogeneous <ins>Link</ins> Prediction (CI-Path)





## Getting Started

### Requirements
- Please follow the links below to install PyTorch and DGL with proper CUDA versions
    - PyTorch https://pytorch.org/
    - DGL https://www.dgl.ai/pages/start.html

- Then install packages by running the line below
```bash
pip install -r requirements.txt
```

- Our code has been tested with
    - Python = 3.10.6
    - PyTorch = 1.12.1
    - DGL = 0.9.1


### Datasets
To citation dataset used in the paper is under `datasets/`. The dataset is after augmentaion, so edges of type `likes` have been added. Similarly for the synthetic dataset. For details of the datasets, please refer to the paper. 

You may also add your favourite datasets by modifying the `load_dataset` function in `dataset_processing.py`.

### GNN Model
We implement the `RGCN` model on heterogeneous graph in `model.py`. A pre-trained model checkpoint is stored in `saved_models/`.


### Explainer Usage
- Run CI-Path to explain trained GNN models 
  - A simple example is shown below
  ```bash
    python pagelink.py --dataset_name=aug_citation --save_explanation
  ```

  - Hyperparameters maybe specified in the `.yaml` file and pass to the script using the `--config_path` argument.
  ```bash
    python pagelink.py --dataset_name=synthetic --config_path=config.yaml --save_explanation
  ```

- Train new GNNs for explanation
  - Run `train_linkpred.py` as the examples below
    ```bash
    python train_linkpred.py --dataset_name=aug_citation --save_model --emb_dim=128 --hidden_dim=128 --out_dim=128
    ```

    ```bash
    python train_linkpred.py --dataset_name=synthetic --save_model --emb_dim=64 --hidden_dim=64 --out_dim=64
    ```

- Run baselines 
    - A simple example is shown below, replace `method` with `gnnexplainer-link` or `pgexplainer-link`.
    ```bash
    python baselines/{method}.py --dataset_name=aug_citation
    ```




## Results

### Quantitative
- Evaluate saved CI-Path explanations
```bash
python eval_explanations.py --dataset_name=synthetic --emb_dim=64 --hidden_dim=64 --out_dim=64 --eval_explainer_names=pagelink
```

**Note**: As exact reproducibility is not guaranteed with PyTorch even with identical random seed
(See https://pytorch.org/docs/stable/notes/randomness.html), the results may be slightly off from the paper.






<!-- #endregion -->
