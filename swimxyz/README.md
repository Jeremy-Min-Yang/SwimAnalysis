# SwimXYZ class project

Side project, separate from Swimline: train a swim-analysis model on the synthetic
[SwimXYZ](https://g-fiche.github.io/research-pages/swimxyz/) dataset (CC BY 4.0), then test it on real footage
and compare with Swimline.

## Main metrics

| Metric | Ground truth in SwimXYZ | Real-footage check |
|---|---|---|
| Stroke phase segmentation (per frame, per arm) | derived from true 3D joints | visual / hand-labelled frames |
| Index of Coordination (IdC) | derived from the phases | qualitative |
| Speed (m/s) | global 3D pelvis track (if present — verify in notebook 01) | `data/labels.csv`: pool length ÷ (finish − start) |
| Distance per stroke (DPS) | speed ÷ stroke frequency | pool length ÷ stroke count |
| Stroke index (SI = speed × DPS) | from the two above | from the two above |

## Notebooks (Google Colab)

1. `01_explore_labels.ipynb`: download one label zip (~1.6 GB) and inspect its format
2. `02_preprocess.ipynb`: parse all labels into a compact `.npz` on Drive *(next)*
3. `03_train.ipynb`: models and experiments *(later)*
4. `04_real_footage.ipynb`: run on real clips and compare with Swimline *(later)*

Only the label zips (6.7 GB total) are needed; the 300 GB of video is not.

## Citation

Fiche, Sevestre, Gonzalez-Barral, Leglaive, Séguier. *SwimXYZ: A large-scale dataset of synthetic swimming motions
and videos.* ACM SIGGRAPH MIG 2023. doi:10.1145/3623264.3624440
