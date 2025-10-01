# **A Scaling Law-Based Sampling Framework for Sample Sufficiency in Geoscience**

This repository contains the official code for the manuscript, "A Scaling Law-Based Sampling Framework for Sample Sufficiency in Geoscience." This work introduces ScaL-Frame, a computational framework that leverages predictable Scaling Laws to transform sample collection from an ad-hoc process into a data-driven science.

**Note:** This work is currently under consideration and has not yet been published.

## **Overview**

The efficient acquisition of training data is a fundamental challenge in machine learning, particularly in scientific domains like geoscience where sampling is costly and data exhibit complex spatial structures. Our work establishes that a predictable Scaling Law can govern model performance in geospatial applications, contingent upon a **Spatially Random Annotation (SRA)** protocol that mitigates spatial biases.

This repository provides the code to reproduce our key findings, including:

* The empirical validation that the SRA protocol (simulated as Random Point Sampling) is a prerequisite for predictable learning curves.  
* The implementation of experiments that demonstrate ScaL-Frame's ability to provide reliable early-stage forecasts.  
* The analysis quantifying the prohibitive cost of marginal accuracy gains and the performance ceilings of low-cost pseudo-label alternatives.

## **Repository Structure**

.  
├── exp\_acc\_scaling\_law.py    \# Main script to run all scaling law experiments  
├── proc\_samples.py           \# Functions for loading and preprocessing datasets  
├── utils.py                  \# Helper functions for parallel processing  
└── README.md                 \# This README file

## **Installation**

1. Clone the repository:  
   git clone \[https://github.com/\](https://github.com/)\[OptimalConvergence\]/\[ScaL-Frame\].git  
   cd ScaL-Frame

2. Create a Python environment and install the required dependencies. We recommend using Conda:  
   conda create \-n scal-frame python=3.9  
   conda activate scal-frame

3. Install the necessary packages using pip:  
   pip install numpy pandas xgboost scikit-learn tqdm geopandas

## **Data**

The datasets used in this study (FAST, LUCAS, CoastTrain) are publicly available. You can download the prepared data from our repository on Figshare:

[**https://figshare.com/s/6cdcbe71b53df163f3d2**](https://figshare.com/s/6cdcbe71b53df163f3d2)

After downloading, please place the data in a directory structure that the code expects. Based on proc\_samples.py, the expected path is ../../assets/LCSamples. This means you should create an assets directory at the same level as your repository folder:

\- project\_directory/  
  ├── assets/  
  │   └── LCSamples/  
  │       ├── FAST/  
  │       ├── LUCAS/  
  │       └── CoastTrain/  
  └── ScaL-Frame/  
      ├── exp\_acc\_scaling\_law.py  
      └── ...

## **Usage**

The main experimental script is exp\_acc\_scaling\_law.py. You can run all experiments from the command line.

#### **Key Arguments:**

* \-samples: The dataset to use (fast, coasttrain, lucas).  
* \-model: The machine learning model to train (rf, xgboost, knn, nn).  
* \-sampling: The sampling protocol to simulate (rps for Random Point Sampling, biased for spatially dependent methods).  
* \-season: The season to filter data for (e.g., summer, winter).  
* \-level: The classification level (e.g., 1).  
* \-repeat: A flag to repeat experiments multiple times for statistical robustness.

### **Example Commands**

#### **1\. Reproduce the Main Scaling Law Experiment (Random Point Sampling)**

This command runs the experiment for the FAST dataset using a Random Forest model and the Random Point Sampling (RPS) protocol.

python exp\_acc\_scaling\_law.py \-samples fast \-model rf \-sampling rps \-season summer \-level 1 \-n\_workers 10

#### **2\. Compare Spatially Biased vs. Random Sampling**

This command runs the experiments for all spatially dependent sampling protocols (Random Grid and Sequential Grid Sampling).

python exp\_acc\_scaling\_law.py \-samples fast \-model rf \-sampling biased \-season summer \-level 1 \-n\_workers 10

#### **3\. Evaluate the Impact of Model and Feature Complexity**

This command runs the sensitivity analyses described in the paper.

python exp\_acc\_scaling\_law.py \-samples fast \-model rf \-sampling eval\_impact \-season summer \-level 1 \-n\_workers 10

Results for each experiment will be saved as a .csv file in the ./outs/acc\_sl/ directory.

## **Citation**

This work is currently under review. If you use this code or our findings in your research, we would appreciate a citation to our manuscript. Please cite the repository for now, and we will update this section with the full citation upon publication.

## **License**

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## **Contact**

For any questions regarding the code or the paper, please contact Yuanhong Liao at liaoyh21@mails.tsinghua.edu.cn.
