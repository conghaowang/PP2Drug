# PP2Drug

Code repository for the paper *Pharmacophore-constrained de novo drug design with diffusion bridge*. The paper is under screening on arXiv. 

## Environment setup

This work is built upon the following packages:
- rdkit==2022.3.5
- openbabel==3.1.1.1
- PyTorch==2.0.1
- PyTorch-Geometric==2.5.3

A conda environment can be built with the config file `environment.yml`
```shell
conda env create --file=environment.yml
```

## Training models from scratch

### Data preprocessing

Download the CrossDocked2020 v1.3 dataset from [here](https://github.com/gnina/models/tree/master/data/CrossDocked2020). 

Following [TargetDiff](https://github.com/guanjq/targetdiff), we firstly filter for ligand-receptor complexes with intimate binding poses (RMSE $<$ 0.1 Å):

```python
cd src/data_processing
python clean_crossdocked.py --source data/CrossDocked2020 --dest data/crossdocked_v1.1_rmsd1.0 --rmsd_thr 1.0
```

Then prepare pytorch geometric dataset with
```python
python paired_data.py --aromatic --filtering
```

### Train a model

Run `main.py`. Specific configurations like dataset path can be modified in the config file. 

```python
cd src
python main.py --config config/vp_bridge_egnn.yml --gpu 0
```

## Sampling with developed models

Trained model checkpoints are available [here](https://drive.google.com/drive/folders/1MPvCAgRRNiAWf1K15c1f_f9j_dYAdE0-?usp=sharing).

### Unconditional generation

```python
python uncond_sample.py \
    --config config/uncond_vp_bridge_egnn.yml \
    --ckpt ../model_ckpt/aromatic_vp_egnn_20240819.ckpt \
    --save ../generation_res \
    --gpu 0 \
    --num_samples 10000 \
    --number_samples=10000
```

### Pharmacophore-constrained generation

#### Batch generation for the test dataset

```python
python sample.py \
    --config ../config/vp_bridge_egnn.yml \
    --ckpt ../model_ckpt/aromatic_vp_egnn_20240819.ckpt \
    --save ../generation_res \
    --gpu 0 \
    --num_samples 10000 \
    --number_samples=10000
```

#### Generation for specific pharmacophore hypotheses

Refer to `evaluation/ligand_based_design_examples.ipynb` or `evaluation/structure_based_design_examples.ipynb` to design a pharmacophore hypothesis.

```python
cd evaluation
python generate_specific_site.py \
    --ligand 1oty_A_rec_3occ_dih_lig_tt_docked_18
    --num_samples 10 \
    --config ../config/vp_bridge_egnn.yml \
    --ckpt ../../model_ckpt/aromatic_vp_egnn_20240819.ckpt \
    --save structure_based \
    --bridge_type vp
    --gpu 0 \
    --number_samples=10000
```

## Evaluation

### Unconditional generation

Assess validity, novelty, uniqueness, SA and QED of the generated molecules. SA and QED score distributions will be saved in a pickle file. Refer to `notebooks/SA_QED.ipynb` to visualize the distributions. 

```python
cd evaluation
python eval_uncond.py --root ../../generation_res --aromatic
```

### Pharmacophore matching evaluation

Assess if the generated molecules align with the prior pharmacophore hypothesis. The matching scores will be stored under `evaluation/<root_path>/<bridge_type>/<ligand_name>`

```python
cd evaluation
python pp_matching_specifc_site.py \
    --config ../config/vp_bridge_egnn.yml
    --root_path structure_based/
    --bridge_type vp
    --aromatic
    --ligand_name 1oty_A_rec_3occ_dih_lig_tt_docked_18
```

### Molecular docking

Assess the binding affinity of generated molecules when docking to the receptor protein. This experiment is conducted using [GNINA v1.1](https://github.com/gnina/gnina/releases/download/v1.1/gnina). Please refer to [GNINA official repository](https://github.com/gnina/gnina) if you want to install other versions. 

```python
cd evaluation
python dock_specifc_site.py \
    --root structure_based/
    --bridge_type vp
    --aromatic
    --ligand 1oty_A_rec_3occ_dih_lig_tt_docked_18
    --gpu 0
```

The docking results will be stored under `evaluation/<root>/<bridge_type>/<ligand>`. Please refer to `evaluation/BA_specifc_site.ipynb` for extracting the binding affinities. 