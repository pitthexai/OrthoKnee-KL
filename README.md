
# Data-Efficient and Trustworthy Adaptation of Vision Foundation Models for Knee Osteoarthritis Classification from Plain Radiographs

## Table of Contents

- [Abstract](#abstract)
- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Publications](#publications)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)



### Abstract

<p align="justify">
This repository contains the implementation accompanying our study on the data-efficient and trustworthy adaptation of vision and vision–language foundation models for knee osteoarthritis (OA) classification from plain radiographs. Using 8,950 knee radiographs from the Osteoarthritis Initiative (OAI), we systematically evaluate four pretrained models: CLIP, DINOv2, Qwen3-VL, and MedSigLIP.

The experiments investigate three dimensions: (1) adaptation strategy (frozen visual representations versus end-to-end visual fine-tuning), (2) training-data regime (20%, 50%, and 100% of the training data), and (3) input modality (radiographs alone versus radiographs combined with clinical metadata). Model performance is evaluated using accuracy, macro-F1, and ROC-AUC, followed by reliability analyses involving uncertainty estimation, calibration, risk–coverage assessment, and representation visualization.

With full supervision and metadata, Qwen3-VL achieved the highest macro-F1 (0.8626), while MedSigLIP achieved the highest ROC-AUC (0.9367). The results highlight model-dependent differences in adaptation, data efficiency, and the contribution of clinical metadata.

</p>



### Highlights

- Evaluation of CLIP and DINOv2 Vision Foundation Models.
- Frozen Feature Extraction strategy.
- End-to-End Fine-Tuning strategy.
- Image-only and Multimodal Learning strategies.
- Entire-data and Limited-data experimental settings.
- Trustworthy AI evaluation with uncertainty and calibration analysis.
- Risk-coverage assessment and embedding visualization.
- Fully reproducible experimental pipeline.



### Repository Structure


<p align="justify">
This repository is organized into modular directories to facilitate reproducibility, readability, and future extension.

</p>

<p align="justify"><strong>Code:</strong> This directory contains all Python scripts and configuration files used for data preprocessing, model training, evaluation, and trustworthy AI analyses.</p>
<p align="justify"><strong>Dataset:</strong> This directory contains dataset descriptions, clinical metadata, preprocessing information, and data split files associated with the Osteoarthritis Initiative (OAI) dataset. Raw radiographs are not included due to data-sharing restrictions.</p>
<p align="justify"><strong>Figures:</strong> This directory contains the figures and visualizations generated for the study, including the proposed pipeline, performance plots, uncertainty analyses, calibration diagrams, risk-coverage curves, and embedding visualizations.</p>
<p align="justify"><strong>Models:</strong> This directory includes trained model checkpoints and configuration files for the CLIP- and DINOv2-based experiments.</p>





### Pipeline

<p align="justify">
The proposed computational framework provides a comprehensive and reproducible pipeline for automated knee osteoarthritis (OA) classification from plain radiographs using pretrained Vision Foundation Models (VFMs). The workflow begins with cohort definition and data preparation, followed by region-of-interest (ROI) extraction and standardized image preprocessing to generate consistent model inputs. Binary knee OA classification (KL < 2 versus KL ≥ 2) is then performed using two state-of-the-art pretrained foundation models, CLIP and DINOv2, under both frozen feature extraction and end-to-end fine-tuning strategies. In addition to image-only learning, the framework supports multimodal learning by integrating radiographic images with structured clinical metadata to investigate the complementary value of clinical information for disease classification.
</p>

<p align="justify">
To systematically evaluate model generalizability under realistic clinical scenarios, experiments are conducted using both entire-data and limited-data training settings with validation-based model selection. Beyond conventional classification performance, the proposed framework incorporates a comprehensive trustworthy AI evaluation pipeline, including predictive uncertainty estimation, calibration analysis, risk-coverage assessment, bootstrap confidence intervals, and feature embedding visualization. Together, these components provide a transparent, reliable, and fully reproducible framework for investigating the adaptation of Vision Foundation Models to musculoskeletal imaging applications.

</p>

<p align="center">
<img src="Figures/ISBI_pipeline_V.1..png" width="800"/>
</p>




### Dataset

<p align="justify">

This study utilizes plain knee radiographs obtained from the Osteoarthritis Initiative (OAI) database.


</p>



### Publications



### Acknowledgments



### Citation

<p align="center">
  <a href="https://pitthexai.github.io/index.html" target="_blank">
    <img src="Figures/Pitthexai_QR.jpg" alt="Support QR Code" width="200"/>
  </a><br/>
  <b>Pitt Health + Explainable AI (Pitt HexAI) Research Laboratory</b>
</p>
