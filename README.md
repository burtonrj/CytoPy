# CytoPy: a cytometry analysis framework for Python
<img src="https://github.com/burtonrj/CytoPy/blob/master/logo.png" width="100" height="100">

In recent years there has been an explosion in Cytometry data analysis tools in the open source scientific community.
This expansion is looking to soon replace traditional methods such as manual gating with sophisticated automated algorithms.

Although exciting, this can be daunting to those from a traditional immunology background and those 
that are new to programming. Additionally, current tools have a loose structure in the steps taken in analysis, 
resulting in large custom scripts, poor reproducibility, and insufficient data management.

CytoPy was created to address these issues. It was created with the general philosophy that given some 
cytometry data and a clinical/experimental endpoint, we wish to find what properties seperate groups (e.g. what cell populations
are important for identifying a disease? What phenotypes are changing in response to a stimulus? etc). 
The pipeline itself is centered around a MongoDB database, built in  the Python programming language, 
and designed with 'low code' as a core aim, greatly 
simplifying cytometry analysis. We can break it all down into the following steps that can be completed 
in just a few lines of code:

1. Data uploading
2. Pre-processing
3. Batch-effect analysis
4. Supervised cell classification 
5. High-dimensional clustering
6. Feature extraction, selection, and description

You will notice that we perform both supervised cell classification and high-dimensional clustering.
Supervised classification being training samples, gated according to some 'gating strategy', being used
to train a classifier. Alternatively high-dimensional clustering (by PhenoGraph or FlowSOM) involves clustering 
cells in a completely unbiased fashion. CytoPy provides access to both methodologies as we observe 
that both have benefits and failings.

Alternatively, after step 3 the user can choose to perform 'cell level association'. 
What does this mean? Rather than classifying cells according to their 'natural' clustering 
or some predefined 'gating strategy' we simply label each individual cell with the meta data of interest 
(experimental endpoint or patient disease state for example) and then fit a classifier such
that it can accurately predict the endpoint given cell-level information and then extract from 
the classifier the feature weights, therefore allowing us to deduce cellular phenotypes of importance.

All above steps are discussed in the manuscript and tutorials linked at the bottom of the page.

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

## Tutorials and examples

* Analysing a novel Sepsis dataset [Manuscript: ####]
* Reproducing results from FlowCAP Challenges
* Reproducing results from ...
* Detailed tutorials:
    1. Loading data into CytoPy
    2. Pre-processing steps
    3. Visualising and quantifying batch effect
    4. Selecting training data
    5. Supervised cell classification and a comparison of method
    6. High dimensional clustering with PhenoGraph and FlowSOM
    7. Meta-clustering and visualisation
    8. Feature extraction and selection
    9. Classification at the single-cell level


