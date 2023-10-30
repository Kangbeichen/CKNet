# CKNet

This is the implementation of [Bridging the Gap: Cross-modal Knowledge Driven
Network for Radiology Report Generation]() at BIBM-2023.



## Requirements

- `torch==1.5.1`
- `torchvision==0.6.1`
- `opencv-python==4.4.0.42`



## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you need to download the dataset and put the files in `data/iu_xray`.

For `MIMIC-CXR`, you need to download the dataset and put the files in `data/mimic_cxr`.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.
