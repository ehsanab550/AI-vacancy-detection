# AI-vacancy-detection
<img width="708" height="390" alt="image" src="https://github.com/user-attachments/assets/420616b4-5815-4b29-8893-abf40c3b36ca" />

AI model for detecting vacancies in 2D materials
# AI Vacancy Detection in 2D Materials


A machine learning-based approach for detecting and analyzing vacancies in 2D materials using experimental STM images.

## Overview

This project provides a computational tool for automatically identifying and characterizing defects in 2D materials through image analysis and machine learning. The system processes experimental scanning tunneling microscopy (STM) images and predicts defect coordinates with high accuracy.

## Features

- **Multi-material support**: MoS₂, graphene, and phosphorene
- **Synthetic data generation**: Automatic creation of training datasets
- **High-accuracy prediction**: Machine learning model for defect detection
- **Experimental image processing**: Advanced image analysis capabilities

## Installation

### Prerequisites

This project requires the following Python library:

--- conda install numpy scipy matplotlib

--- pip install pybinding

### Usage
### Input setting

- In AI_vacancy.ipynb, in "CONFIGURATION SECTION" set the input parameters

#### For Experimental Image Analysis
- **To achieve high-accuracy predictions from experimental STM images, follow these steps**:

  1) Set System Dimensions: Define the physical size of the 2D material layer in nanometers as an input parameter.

  2) Use High-Resolution Images: Ensure input images have high resolution (>120 DPI) for optimal feature extraction.

  3) Select Material: In AI_vacancy.ipynb, specify the material (MoS₂, graphene, or phosphorene).

  4) For unsupported materials: Define the crystal lattice in PyBinding (lattice constants and atom coordinates)

  5) Generate Training Data: The code automatically processes your experimental image and generates synthetic training data with diverse defects.

  6) Predict Defect Coordinates: The ML model combines experimental features with synthetic training to predict precise defect coordinates.

### Main Scripts

  AI_vacancy.ipynb: Main code for vacancy detection and analysis

  Plotting utilities: Use the provided plotting scripts for visualization.

### Supported Materials

-MoS₂ (Molybdenum Disulfide)

-Graphene

-Phosphorene

-Extensible to other 2D materials

### Methodology

> Image Preprocessing: Convert experimental images to grayscale and enhance features

> Synthetic Data Generation: Create realistic defect simulations using PyBinding (2Dmaterials.py)

> Feature Extraction: Analyze texture, contrast, and structural patterns

> Machine Learning: Random Forest model trained on synthetic-experimental feature combinations

> Coordinate Prediction: Precise defect localization and characterization

### Contributing

To add support for new materials:

  ++ Define the crystal lattice in PyBinding

  ++ Add material-specific parameters to the configuration

  ++ Generate appropriate synthetic training data
