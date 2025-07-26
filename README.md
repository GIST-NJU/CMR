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
├── results
│   ├── errors                          # Failures and Faults
│   ├── features                        # Extracted features for RQ3.2
│   ├── predictions                     # Predictions of DNNs on source and followup test inputs
│   └── SelfOracle
|   |   ├── COCO_threshold.txt
|   |   ├── COCO_VAE.pth
|   |   ├── COCO_validity.npy
|   |   ├── ...
├── RQs.ipynb
└── src
    ├── generate_followup.py
    ├── __init__.py
    ├── mr_utils.py
    ├── predict
    ├── predict.py
    ├── selforacle
    └── selforacle.py
```

## ⚙️Requirements

    python==3.8.18
    ipykernel==6.26.0
    jupyter_client==7.3.4
    jupyter_core==5.5.1
    torch==2.0.1+cu118
    torchvision==0.15.2+cu118
    sklearn==1.3.2
    cv2==4.7.0
    numpy==1.23.5
    pandas==1.4.3

## 📦Experimental Subjects
| Models & Datasets | Download Link | Local Destination Path |
|---|---|---|
| *DNN Under Test* | [Download](https://www.dropbox.com/scl/fo/x9et5salo528e8inh2999/ANrIfVqVdQqvUrVFzkKiiu8?rlkey=hetb4y6f7hwtqzeay9nwz1fpn&dl=0) | `models/` |
| *Source Test Input Set* | [Download](https://www.dropbox.com/scl/fo/zfqodjegi4wh0n04mlh7d/AHd8P5BftYNTmszXqygRudE?rlkey=wowwl40k8hr2mmy3shyo420zn&dl=0) | `data/` |

For Caltech256: After downloading the source test input set to the local destination path, extract it with:
```bash
tar -xf data/caltech256/256_ObjectCategories.tar -C data/caltech256
```

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
    Output: the VAE model, threshold, and validity of followup inputs are stored in `results/SelfORacle/`. For example, `COCO_VAE.pth`, `COCO_threshold.txt`, `COCO_validity.pth`
3. **Make predictions**
    ```bash
    python predict.py --dataset COCO --source True # test source
    python predict.py --dataset COCO --source False --strength 2 # test followup
    ```
    Output: the predictions of the model under test on source are followup inputs are saved in `results/predictions/dataset/`. For example, the `results/predictions/COCO/COCO_MLD_source.csv` and `results/predictions/COCO/COCO_MLD_followup.npy`
4. **Compute the results for RQs**
    ```bash
    RQs.ipynb
    ```
    Output: obtain data for all RQs. Table data is printed directly within the book, while figures are saved  in `figures/`
