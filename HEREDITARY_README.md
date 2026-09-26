# mllabiome 0.1.0rc123

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

| Field | Value |
|---|---|
| Name | mllabiome |
| Version | 0.1.0rc123 |
| Type | Software or library |
| Owner partner and contact | Radboud University Medical Center (RUMC), CMG-GUTS |
| License | Apache-2.0 |
| Repository and DOI | https://github.com/CMG-GUTS/mllabiome — **software-release DOI to be added when archived** |
| Release date | **[to be added, date of the tagged 0.1.0rc123 release]** |
| Related task or deliverable | HEREDITARY T4.3, Self-supervision-powered multimodal data learning; related to D4.3, Learning models and spatio-temporal harmonization |
| How to cite | **[software DOI / accompanying methods publication to be added when available]** |

## 2. Intended use

**Primary intended use.**  
`mllabiome` is a research software library for leakage-controlled machine-learning analysis of microbiome data. It supports evaluation and comparison of microbiota profile data representations, machine-learning algorithms, ensemble models and multimodal models using nested and cohort-aware validation strategies. It also provides model interpretation, robustness analysis, statistical comparison and research reporting functionality.

The software is intended to support research questions involving predictive modelling of microbiome-derived features, comparison of alternative microbiome representations and modelling strategies, assessment of model generalizability across subjects or cohorts, multimodal data integration, and exploratory generation of biological hypotheses.

The software operates on processed microbiome feature or abundance tables and associated research metadata. It is not intended as a raw sequencing-processing pipeline.

**Intended users.**  
Biomedical researchers, microbiome researchers, bioinformaticians, computational biologists, biostatisticians, machine-learning researchers and data scientists with appropriate expertise in the analysis and interpretation of biomedical research data.

**Out-of-scope uses.**  
This release must not be used for:

- clinical diagnosis, prognosis, treatment, monitoring or other clinical management of individual patients;
- decisions about individuals, including decisions concerning insurance, employment or access to healthcare;
- direct use as a medical device or clinical decision-support system;
- re-identification of individuals or linkage of data for the purpose of identifying research participants;
- use of model predictions or feature-attribution results as evidence of causal biological relationships;
- populations, clinical settings, sequencing technologies, preprocessing procedures or data types for which the resulting model has not been independently validated;
- processing of personal or patient data without an appropriate lawful basis, ethics approval where applicable, institutional authorization and appropriate technical and organizational safeguards.

## 3. Data provenance

This release is a software library and does not itself constitute a research dataset.

| Dataset | Source partner | Type | Population | Ethics approval or access conditions |
|---|---|---|---|---|
| No study dataset distributed as part of this software release | Not applicable | Software release | Not applicable | Not applicable |

The `mllabiome` software is designed to operate on user-supplied processed microbiome and associated metadata. Responsibility for the provenance, lawful processing, ethical approval, consent, access restrictions and de-identification status of input datasets remains with the user and the institution responsible for the corresponding research project.

This software release should contain no personal data, pseudonymized research-participant records, direct or indirect participant identifiers, or pseudonymization keys.

Any example or test data distributed with the release should be synthetic, generated, or otherwise suitable for unrestricted public distribution and should be reviewed separately before release.

## 4. Method

- **Algorithm or architecture.**  
  `mllabiome` is a Python-based machine-learning framework for microbiome research rather than a single predictive algorithm. It supports systematic evaluation of microbiota profile data representations (MPDRs), microbiota profile modelling algorithms (MPMAs), ensemble modelling, multimodal integration, nested cross-validation, grouped and subject-aware splitting, leave-one-dataset-out evaluation, leakage-controlled preprocessing, statistical comparison, explainable AI and robustness analysis.

  MPDR denotes **Microbiota Profile Data Representation**.

  MPMA denotes **Microbiota Profile Modelling Algorithm**. MPMA-B denotes a **base microbiota profile modelling algorithm**, while MPMA-E denotes an **ensemble microbiota profile modelling algorithm**.

  Preprocessing and model-selection operations are designed to be fitted within the appropriate training partitions so that held-out observations do not influence the workflow used to predict them.

- **Training setting.**  
  Not applicable at the release level. `mllabiome` is a software library and no pretrained clinical prediction model is distributed as part of this release. Models are trained by users on datasets supplied for individual research studies.

- **Privacy techniques.**  
  No dedicated differential-privacy, secure-aggregation or privacy-preserving federated-learning mechanism is implemented as a privacy guarantee in this software release. Privacy and data-governance requirements therefore depend on the deployment environment and the datasets supplied by users.

## 5. Performance

- **Evaluation data.**  
  Not applicable to the software release as a whole. `mllabiome` is a general-purpose research analysis framework rather than a trained prediction model. Predictive performance depends on the dataset, phenotype, microbiome representation, modelling algorithm, validation design and study population used in a particular analysis.

- **Metrics and results.**  
  No single AUC, accuracy, regression score or other predictive-performance value characterizes this software release. The library supports calculation and reporting of appropriate classification and regression metrics within leakage-controlled validation procedures. Quantitative performance claims should be reported separately for each research dataset and experimental protocol.

- **Results by subgroup.**  
  Not applicable to the software release itself. Where subgroup analyses are scientifically and statistically appropriate, they must be performed and reported for the specific study using the software.

No clinical-performance claim is made for `mllabiome 0.1.0rc123`.

## 6. Known limitations and risks

- **Generalizability.**  
  Results obtained using `mllabiome` are dependent on the population, cohort, sampling procedure, sequencing or profiling technology, bioinformatic preprocessing, taxonomic or functional representation, phenotype definition and modelling configuration used in the corresponding study. Performance observed in one dataset or population must not be assumed to generalize to another population, site or clinical setting without independent validation.

  Microbiome datasets may exhibit substantial inter-cohort heterogeneity caused by geography, demographics, sample collection, storage, sequencing platforms, laboratory protocols, bioinformatic processing and other technical or biological factors.

- **Known biases and confounders.**  
  Microbiome predictive models can be affected by confounding variables including age, sex, medication, diet, geography, clinical site, sequencing batch, sample-processing protocol and disease-associated treatment. Small sample sizes combined with high-dimensional feature spaces may lead to unstable estimates and overfitting.

  Microbiome abundance data are compositional, sparse and strongly correlated. Predictive feature importance or explainability results therefore must not automatically be interpreted as independent biological effects or causal biomarkers.

  Dataset imbalance, missing data, repeated measurements, cohort structure and inappropriate validation design may produce misleading performance estimates if not handled correctly.

- **Failure modes.**  
  Results may be unreliable when input tables or metadata are incorrectly aligned, sample identifiers are duplicated or inconsistent, labels are incorrect, inappropriate grouping variables are supplied, sample sizes are insufficient, classes are highly imbalanced, external cohorts differ substantially from the training population, or preprocessing assumptions are violated.

  Computational failures may also arise from incompatible or missing optional software dependencies, insufficient memory or compute resources, malformed configuration files, unsupported data structures, or failures of external software integrations.

  Explainability methods such as SHAP, LIME, permutation importance and accumulated local effects describe properties of fitted predictive models and must not be interpreted as demonstrating biological causation.

- **Privacy risks.**  
  No trained patient-level model or research-participant dataset is distributed as part of this software release; therefore membership-inference or model-inversion testing is not applicable to the release itself.

  Models trained by users on sensitive datasets may nevertheless introduce privacy risks. If trained models, predictions, embeddings, feature-attribution outputs or other derived artefacts are shared, the responsible research team must assess disclosure and re-identification risks separately.

## 7. Responsible use and reporting

- **Terms of use.**  
  The software is released under the Apache License 2.0 for research and software-development use, subject to the conditions of that license.

  The open-source software license does not constitute regulatory approval, medical-device certification, validation for clinical practice, authorization to process personal data, or permission to use restricted datasets.

  Users are responsible for ensuring that any data processed using `mllabiome` are handled in accordance with applicable ethics approvals, informed-consent conditions, institutional policies, data-use agreements and data-protection legislation.

  Results generated by the software should be reported together with sufficient information about the dataset, preprocessing, validation strategy, model-selection procedure, software version and analysis configuration to permit scientific interpretation and reproducibility.

- **Contact for misuse or vulnerability reports.**  
  **Software project maintainer: Agata Polejowska**

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.