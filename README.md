# How Composite Metamorphic Relations Enhance Test Effectiveness of DNN Testing: An Empirical Study

This repository provides code and data for our paper.

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

## 📦Resources

| Resources | Download Link | Local Destination Path |
|---|---|---|
| DNN Under Test | [Download](https://www.dropbox.com/scl/fo/x9et5salo528e8inh2999/ANrIfVqVdQqvUrVFzkKiiu8?rlkey=hetb4y6f7hwtqzeay9nwz1fpn&dl=0) | `models/` |
| Source Test Input Set | [Download](https://www.dropbox.com/scl/fo/zfqodjegi4wh0n04mlh7d/AHd8P5BftYNTmszXqygRudE?rlkey=wowwl40k8hr2mmy3shyo420zn&dl=0) | `data/` |
| Predictions | [Download](https://www.dropbox.com/scl/fo/moicow2jgo0q05pgmi6gq/ACkrFH2xQKBdzJ-VqlvSSQE?rlkey=mw75vfwv9gz7pux9cleutryuv&dl=0) | `results/predictions/` |
| SelfOracle for Validity| [Download](https://www.dropbox.com/scl/fo/0cicv66v6a5eex6rjbkpn/AGgBgdneMProMpCZIG5Riq4?rlkey=qpbthrzlz56sc3bgthz3609nk&dl=0) | `results/SelfOracle/` |
| Failure / Fault | [Download](https://www.dropbox.com/scl/fo/djz19v7z6zevl7rl0gewr/ALmhuzP_NqHgkvYJsxZ1E1A?rlkey=0h1zhzqovcerion5egpmlmtew&dl=0) | `results/errors/` |
| Extracted Features for RQ3.2| [Download](https://www.dropbox.com/scl/fo/5faddj9zfczyaw4lr33rg/AMO1Wg_lhfuUaAmr2Nt5Xa0?rlkey=2wuru62a9tf5yhjsq0slavsy8&dl=0) | `results/features/` |

All MRs for composition are implemented in `src/mr_utils.py`

## 🛠️Reproducing Experiment

1. **Generate follow-up test inputs**
    ```bash
    python generate_followup.py --dataset MNIST --strength 2
2. Identify valid test inputs
    ```bash
    python selforacle.py --dataset MNIST --strength 2
3. Make predictions
    ```bash
    python predict.py --dataset MNIST --source # test source
    python predict.py --dataset MNIST --source False --strength 2 # test followup
4. Compute the results for RQs
    ```bash
    RQs.ipynb
