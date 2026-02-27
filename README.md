# AI Vacancy Detection in 2D Materials

<img width="708" height="390" alt="STM image showing vacancies in 2D material" src="https://github.com/user-attachments/assets/420616b4-5815-4b29-8893-abf40c3b36ca" />

[![DOI](https://img.shields.io/badge/DOI-10.1038/s41699--026--00667--4-blue)](https://doi.org/10.1038/s41699-026-00667-4)

A **machine learning‑based approach** for detecting and analyzing vacancies in 2D materials using experimental Scanning Tunneling Microscopy (STM) images.

---

## 📖 Overview

This project provides a computational tool for **automatically identifying and characterizing defects** in 2D materials through image analysis and machine learning. The system processes experimental STM images and predicts defect coordinates with high accuracy.

## ✨ Features

- **Multi‑material support**: MoS₂, graphene, and phosphorene
- **Synthetic data generation**: Automatic creation of realistic training datasets
- **High‑accuracy prediction**: Machine learning model for defect detection
- **Experimental image processing**: Advanced image analysis capabilities

## ⚙️ Installation

### Prerequisites

This project requires the following Python libraries:

#### bash
- conda install numpy scipy matplotlib
- pip install pybinding

## 🚀 Usage
### Input configuration
 In AI_vacancy.ipynb, locate the CONFIGURATION SECTION and set the required parameters.

### For experimental image analysis
To obtain high‑accuracy predictions from experimental STM images, follow these steps:

1. Set system dimensions – Define the physical size of the 2D material layer in nanometers.

2. Use high‑resolution images – Input images should have a resolution >120 DPI for optimal feature extraction.

3. Select material – In AI_vacancy.ipynb, specify one of the supported materials: MoS₂, graphene, or phosphorene.

4. For unsupported materials – Define the crystal lattice in PyBinding (lattice constants and atom coordinates).

5. Generate training data – The code automatically processes your experimental image and generates synthetic training data with diverse defects.

6. Predict defect coordinates – The ML model combines experimental features with synthetic training to predict precise defect locations.

### Main scripts
AI_vacancy.ipynb – Main code for vacancy detection and analysis

. Plotting utilities – Use the provided scripts for visualisation.

## 🧪 Supported Materials
. MoS₂ (Molybdenum Disulfide)

. Graphene

. Phosphorene

. Extensible to other 2D materials

## 🔬 Methodology
- Image Preprocessing – Convert experimental images to grayscale and enhance features
- Synthetic Data Generation – Create realistic defect simulations using PyBinding (2D_materials.py)
- Feature Extraction – Analyse texture, contrast, and structural patterns
- Machine Learning – Random Forest model trained on synthetic‑experimental feature combinations
Coordinate Prediction – Precise defect localisation and characterisation

## 🤝 Contributing
To add support for a new material:

1. Define the crystal lattice in PyBinding

2. Add material‑specific parameters to the configuration

3. Generate appropriate synthetic training data

## 📝 Citation
If you use this code in your research, please cite the following paper:

#### Alibagheri, E. AI‑driven image processing framework for high‑accuracy detection and characterization of vacancies in 2D materials. npj 2D Materials and Applications (2026).
 

@article{alibagheri2026ai,
  title={AI-driven image processing framework for high-accuracy detection and characterization of vacancies in 2D materials},
  author={Alibagheri, E.},
  journal={npj 2D Materials and Applications},
  year={2026},
  doi={10.1038/s41699-026-00667-4}
}
### 📄 License
This project is available for academic use. Please cite the paper if you use the code. For commercial use, please contact the author.

For questions or issues, please open an issue on GitHub.
