# CytoDragon: the simple cytometry analysis pipeline for Python
![Logo](./logo_sm.png)

In recent years there has been an explosion in Cytometry data analysis tools in the open source scientific community.
This expansion is looking to soon replace traditional methods such as manual gating with sophisticated automated algorithms.

Although exciting, this can be daunting to those from a traditional immunology background and those 
that are new to programming. Additionally, current tools have a loose structure in the steps taken in analysis, 
resulting in large custom scripts, poor reproducibility, and insufficient data management.

CytoDragon was created to address these issues. Centered around a MongoDB database, built in 
the Python programming language, and designed with 'low code' as a core aim, CytoDragon greatly 
simplifies cytometry analysis, breaking it down into the following steps that can be completed 
in just a few lines of code:

1. Data uploading
2. Pre-processing
3. Batch-effect analysis
4. Cell classification
5. High-dimensional clustering
6. Feature extraction, selection, and description

## Installation

### Requirements
Python 3x \
MongoDB 

**Python libraries:** 

| Misc | Scipy | Cytometry Stack | Machine Learning | Plotting |
| :--- | :--- | :--- | :--- | :--- |
| mongoengine, anytree, tqdm, xlrd | numpy, pandas, scipy | FlowUtilsPandas, FlowIO |scikit-learn, imblearn, phate, umap, hdbscan, keras, phenograph, minisom | matplotlib, seaborn, scprep, shapely|

Install using pip (not currently active):

`pip install cytodragon`

Or, download source code and run the following from the source directory:

`pip install .`

