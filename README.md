# How Composite Metamorphic Relations Enhance Test Effectiveness of DNN Testing: An Empirical Study

This repository provides code and data for our paper.

```
CMR
├── data                                # Source test inputs
├── dataloaders                         # Dataloads for VOC and COCO
├── figures                             # Figures for RQs
├── followup                            # Followup test inputs
│   ├── COCO
|   |   |── 0
|   |   |── 1
|   |   └── ...
│   └── ...
├── models                              # DNN models under test
├── requirements.txt
├── results
│   ├── errors                          # Failures and Faults
│   ├── features                        # Extracted features for RQ3.2
│   ├── predictions                     # Predictions of DNNs on source and followup test inputs
│   └── SelfOracle
|   |   ├── COCO_threshold.txt          # Threshold corresponding to false alarm
|   |   ├── COCO_VAE.pth                # Trained VAE
|   |   ├── COCO_validity.npy           # Validity of followup test inputs
|   |   └── ...
├── RQs.ipynb                           # Codes for calculating data and drawing figures for RQs
└── src                                 # Codes for generating experimental data
    ├── extract_features.py
    ├── generate_followup.py
    ├── __init__.py
    ├── mr_utils.py
    ├── predict
    ├── predict.py
    ├── selforacle
    └── selforacle.py
```

## ⚙️Requirements

```bash
conda create -n cmr python=3.8.18
conda activate cmr
pip install -r requirements.txt
```

## 📦Experimental Subjects
| Models & Datasets | Download Link | Local Destination Path |
|---|---|---|
| *DNN Under Test* | [Download](https://www.dropbox.com/scl/fo/x9et5salo528e8inh2999/ANrIfVqVdQqvUrVFzkKiiu8?rlkey=hetb4y6f7hwtqzeay9nwz1fpn&dl=0) | `models/` |
| *Source Test Input Set* | [Download](https://www.dropbox.com/scl/fo/zfqodjegi4wh0n04mlh7d/AHd8P5BftYNTmszXqygRudE?rlkey=wowwl40k8hr2mmy3shyo420zn&dl=0) | `data/` |
| *Sampled Followup Test Input Set For Human Validation* | [Download](https://box.nju.edu.cn/d/4185dd7480df45078b21/) | `followup/sample_check_followup/` |

*Cmponent MRs* for composition are implemented in `src/mr_utils.py`

## 📃Experimental Data

- *Predictions*: Outputs of the model on source and follow-up inputs
- *SelfOracle*: The trained VAE models, thresholds, and validity of followup inputs
- *Failure & Fault*: Failures and faults are calculated accordings to *predictions*
- *Extracted Features*: Extracted Features of source and followup inputs for RQ3.2
- *Figures*: Figurs of RQ1, RQ2, RQ3.1, RQ3.2

**Data available at the link below.**

| Resources | Download Link | Local Destination Path |
|---|---|---|
| Predictions | [Download](https://www.dropbox.com/scl/fo/moicow2jgo0q05pgmi6gq/ACkrFH2xQKBdzJ-VqlvSSQE?rlkey=mw75vfwv9gz7pux9cleutryuv&dl=0) | `results/predictions/` |
| SelfOracle | [Download](https://www.dropbox.com/scl/fo/0cicv66v6a5eex6rjbkpn/AGgBgdneMProMpCZIG5Riq4?rlkey=qpbthrzlz56sc3bgthz3609nk&dl=0) | `results/SelfOracle/` |
| Failure & Fault | [Download](https://www.dropbox.com/scl/fo/djz19v7z6zevl7rl0gewr/ALmhuzP_NqHgkvYJsxZ1E1A?rlkey=0h1zhzqovcerion5egpmlmtew&dl=0) | `results/errors/` |
| Extracted Features| [Download](https://www.dropbox.com/scl/fo/5faddj9zfczyaw4lr33rg/AMO1Wg_lhfuUaAmr2Nt5Xa0?rlkey=2wuru62a9tf5yhjsq0slavsy8&dl=0) | `results/features/` |
| Figures | `figures/`| - |


## 🛠️Reproducing Experiment

1. **Generate follow-up test inputs**
    ```bash
    python generate_followup.py --dataset COCO --strength 2
    ```
    Output: Followup inputs are saved in `followup/dataset/cmr`. For example, `followup/COCO/31`
2. **Identify valid test inputs**
    ```bash
    python selforacle.py --dataset COCO
    ```
    Output: The VAE model, threshold, and validity of followup inputs are stored in `results/SelfORacle/`. For example, `COCO_VAE.pth`, `COCO_threshold.txt`, `COCO_validity.pth`
3. **Make predictions**
    ```bash
    python predict.py --dataset COCO # test source
    python predict.py --dataset COCO --followup # test followup
    ```
    Output: The predictions of the model under test on source are followup inputs are saved in `results/predictions/dataset/`. For example, the `results/predictions/COCO/COCO_MLD_source.csv` and `results/predictions/COCO/COCO_MLD_followup.npy`
4. **Count Failure and Fault**
    ```bash
    python count_failure_fault.py -- dataset COCO
    ```
    Output: The failres and faults are saved in `results/errors/`. For example, `failure_COCO_MLD.pkl` and `fault_COCO_MLD.pkl`. Both are stored as dict: key is cmr, and values is the list of violated test groups and the set of fault types, respectively.
5. **Compute the results for RQs**
    ```bash
    python extract_features.py # Extract features for RQ3.2
    RQs.ipynb
    ```
    Output: Extracted features are stored in `results/features/`. For answering RQs, table data is printed directly within the jupyter notebook, while figures are saved  in `figures/`
