[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://scibotscanv3.streamlit.app/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18664634.svg)](https://doi.org/10.5281/zenodo.18664634)


# SciBotScanV3 

## An Explainable Machine Learning Architecture for Altmetric Account Classification on X

DOI: 10.5281/zenodo.18664634

SciBotScan is an explainable supervised machine learning architecture designed to classify altmetric accounts on the X platform (formerly Twitter) as automated (bots) or human-operated accounts, with a specific focus on scientific article dissemination.

The system aims to mitigate distortions in altmetric indicators caused by automated accounts and to support more transparent and reliable evaluation of scientific communication in digital environments.

---

## Overview

SciBotScan implements a probabilistic classification model based on Extreme Gradient Boosting (XGBoost), trained on a manually validated dataset of 13,767 accounts.

The architecture integrates feature engineering, supervised learning, and model interpretability techniques to provide robust and transparent predictions.

The system outputs a probability score indicating the likelihood that a given account is automated.

---

## Key Features

* Supervised classification using XGBoost
* 46 predictive features derived from structural, behavioral, textual, and altmetric attributes
* Probabilistic output (bot likelihood score)
* Decision threshold calibrated for imbalanced data
* SHAP-based interpretability analysis
* Interactive web interface implemented with Streamlit

---

## Architecture

The system follows a modular machine learning pipeline aligned with CRISP-DM principles:

1. Data input and preprocessing
2. Feature engineering and selection
3. Supervised model training and inference
4. Model interpretability using SHAP
5. Deployment through a Streamlit web interface

---

## Dataset

* 13,767 labeled accounts
* 822 bot accounts
* 12,945 human accounts
* More than 67,000 analyzed posts

The dataset was curated through manual validation and cross-referenced with established bot detection research.

---

## Technologies Used

* Python 3.12
* XGBoost
* scikit-learn
* pandas & numpy
* TextBlob (sentiment analysis)
* matplotlib
* Streamlit

---

## Citation

If you use SciBotScanV3 in academic research, please cite:

Pontes, D. P. N., & Maricato, J. M. (2026).  
SciBotScanV3: An Explainable Machine Learning Architecture for Altmetric Account Classification on X (Version 1.0.0) [Software]. Zenodo.  
https://doi.org/10.5281/zenodo.18664634

## License

Specify your license here (e.g., MIT License).

---

## Authors

Danielle P. N. Pontes
PhD in Information Science
University of the State of Amazonas (UEA)

João de Melo Maricato
PhD in Information Science

