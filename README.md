# DPPH Molecular Predictor

A Streamlit research application that predicts an operational DPPH activity category from a connected molecular SMILES string or a batch CSV.

**Endpoint:** Active means reported DPPH IC50 <1000 mg/L; Inactive means IC50 >=1000 mg/L. This is a screening definition and not a universal definition of antioxidant activity.

## Models and reproducibility

The release contains the frozen ExtraTrees count model and generalized-count-Tanimoto SVM from development run15408. Model files are authenticated by SHA256 before loading. The application must reproduce36 reference scores before enabling prediction. The original model parameters and hashes are in `contract.json`.

Training contained2130 compounds. The original263-compound development domain was evaluated using five component folds that were inspected during development. ExtraTrees counts achieved mean balanced accuracy0.722 ±0.039 and ROC-AUC0.789 ±0.058; count SVM achieved0.723 ±0.055 and0.807 ±0.064. These are retrospective development estimates; SD is not a confidence interval.

The app shows both models, their uncalibrated score/margin, bit and count similarities, and five nearest training structures. No model is fitted during application use. Model disagreement is preserved.

## Streamlit Community Cloud deployment

1. Upload the contents of this directory to a GitHub repository, preserving `data/`, `models/`, and `.streamlit/`.
2. In Streamlit Community Cloud, select that repository, branch `main`, and main file `app.py`.
3. In **Advanced settings**, select **Python3.12**. No secrets are required.
4. Deploy. The app loads its included model files automatically; visitors do not upload models.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Local installation

```bash
python3.12 -m venv .venv
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use the virtual environment's Python executable for the last two commands. Run `python verify_release.py` in that environment to check both model hashes and all36 score replays before deployment.

## Input

CSV requires `smiles`; `sample_id` is optional. Limit200 compounds,2MB CSV, and one connected structure per row. The computational size limits are not validated applicability boundaries.

## Interpretation

Scores are not calibrated probabilities. Similarity and agreement do not establish experimental accuracy. The application does not estimate numerical IC50, identify a GCMS library hit, weight peaks, estimate crude-extract efficacy or predict animal outcomes. GCMS example structures remain tentative. Exact connectivity matches with training are flagged.

## Data sources

- Kato et al. Current Issues in Molecular Biology2026;48:12. https://doi.org/10.3390/cimb48010012
- Ghironi et al. Journal of Xenobiotics2025;15:80. https://doi.org/10.3390/jox15030080
- Yamauchi et al. Antioxidants2024;13:309. https://doi.org/10.3390/antiox13030309

Source DOI provenance is retained with the training records. Heterogeneous literature assay conditions and incomplete re-verification of the added compilation remain limitations.

## Version

Research application release1.0.0, prepared17November2026. This release packages existing models and generates no new validation result. A published app URL can be reported as software availability; it is not a substitute for model validation.

## Acknowledgments

The authors gratefully acknowledge the Ministry of Higher Education and Scientific Research (MOHESR) for providing access to the MOHESR High Performance Computing (MOHESR-HPC) platform and computational resources used in this research.
