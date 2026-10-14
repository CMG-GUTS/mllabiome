> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

| Field | Value |
|---|---|
| Name | mllabiome |
| Version | v0.2.0 |
| Type | Software or library |
| Owner partner and contact | Radboud University Medical Center (RUMC), Agata Polejowska, agata.polejowska@radboudumc.nl |
| License | Apache-2.0 |
| Repository and DOI | https://github.com/CMG-GUTS/mllabiome |
| Release date | 14 October 2026 |
| Related task or deliverable | WP4, D4.5, D2.6 |
| How to cite | Agata Polejowska. (2026). *mllabiome* (Version 0.2.0) [Computer software]. https://github.com/CMG-GUTS/mllabiome |

## 2. Intended use

**Primary intended use.** mllabiome is a research-oriented Python framework for developing, evaluating and interpreting machine-learning models using microbiota and additional modalities. It includes multimodal approaches and the use of self-supervised foundation models. It supports reproducible benchmarking, model comparison and biological hypothesis generation.

**Intended users.** Researchers, bioinformaticians, data scientists and machine-learning developers.

**Out-of-scope uses.** This release must not be used for
- clinical diagnosis, prognosis or treatment of individual patients;
- decisions about individuals, such as insurance, employment or access to care;
- re-identification of individuals or linkage with other data sources;
- populations, devices or data types not described in Section 3;
- clinical monitoring of individual patients.

## 3. Data provenance

| Dataset | Source partner | Type (public, synthetic, pseudonymized, anonymized) | Population (size, age, sex, country, condition) | Ethics approval or access conditions |
|---|---|---|---|---|
| LAMPP — Live Assessment of Metagenomics-based tools for human Phenotype Prediction (all six prediction tasks) | Barak et al. (2025), https://doi.org/10.1101/2025.06.12.658885 | Public | Multiple cohorts covering different host phenotypes. Demographics and sample sizes vary by task; see the source publication. | Publicly available benchmark. Ethics approvals are described in the original studies. |
| Gut microbiota and fMRI-derived features | Miranda-Angulo et al. (2024), https://doi.org/10.1016/j.psyneuen.2024.107003 | Public | 88 healthy Colombian men, aged 21–40 years. | Publicly available research data. Original study ethics and data-access conditions apply. |


This repository and the released files contain no personal data, pseudonymized records, identifiers or pseudonymization keys.

## 4. Method

- **Algorithm or architecture.** The framework supports multiple machine-learning algorithms, ensemble methods and pretrained foundation models for unimodal and multimodal analysis. It provides automated model optimization, validation and explainability.
- **Training setting.** Centralized.
- **Privacy techniques.** No dedicated privacy-preserving techniques are documented for the core framework; data are processed locally by default.

## 5. Performance

- **Evaluation data.** Public datasets including all six LAMPP benchmark tasks and data from Miranda-Angulo et al. (2024); evaluation configurations include nested cross-validation and leave-one-dataset-out validation.
- **Metrics and results.** No single performance result applies to the software framework; metrics and results depend on the dataset, prediction task and model configuration.
- **Results by subgroup.** No release-wide demographic subgroup results are reported.

## 6. Known limitations and risks

- **Generalizability.** Model performance depends on the datasets, populations and preprocessing methods used. Generalizability to other cohorts, sequencing protocols and data modalities is not guaranteed.
- **Known biases and confounders.** Microbiome analyses may be affected by cohort differences, batch effects, age, sex, medication and other confounding factors.
- **Failure modes.** Feature explanations do not establish causality.
- **Privacy risks.** The framework processes user-provided data locally. Users are responsible for protecting sensitive data and generated outputs. No trained model weights are distributed. Privacy attacks on optional third-party pretrained models have not been assessed.

## 7. Responsible use and reporting

- **Terms of use.** Released under the Apache-2.0 license. The software is intended for research use only and has not been validated for clinical applications. Users are responsible for ensuring appropriate data protection and compliance with applicable regulations.
- **Contact for misuse or vulnerability reports.** Agata Polejowska, Radboud University Medical Center, agata.polejowska@radboudumc.nl

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.
