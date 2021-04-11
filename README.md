<p align="center">
  <img src="https://github.com/burtonrj/CytoPy/blob/master/logo.png" height="25%" width="25%">
  <h1 align="center">CytoPy: a cytometry analysis framework for Python</h1>
</p>

# Overview

In recent years there has been an explosion in Cytometry data analysis tools in the open source scientific community. This expansion is looking to soon replace traditional methods such as manual gating with sophisticated automated algorithms.

Although exciting, most of the tools and frameworks on offer are implemented in the R programming language and offer little structure and data management for those that are new to cytometry bioinformatics. This is especially difficult for those with limited experience with R and Bioconductor. We offer an alternative solution implemented in Python, a beginner friendly language that prides itself on readable syntax.

The CytoPy framework offers an object orientated design built upon <a href=http://mongoengine.org/>mongoengine</a> for flexible database designs that can incorporate any project, no matter how complex. CytoPy's toolkit populates this database with common data structures to represent cell populations identified in your cytometry data, whilst being algorithm agnostic and encouraging the use and comparison of multiple techniques.

Features we offer are:

* Dynamic central document-based data repository
* Autonomous gating with hyperparameter search and local normalisation to help with tricky batch effects
* Global batch effect correction with the <a href=https://github.com/slowkow/harmonypy>Harmony algoritm</a>
* Supervised classification supporting any classifier in the Scikit-Learn ecosystem
* High dimensional clustering, including but not limited to FlowSOM and Phenograph
* Feature extracting and selection techniques to summarise and interrogate your identified populations of interest
* A range of utilities from sampling methods, common transformations (logicle, arcsine, hyperlog etc), and dimension reduction (including PHATE, UMAP, tSNE, PCA and KernelPCA)

To find out more and for installation instructions, please read our documentation at https://cytopy.readthedocs.io/en/latest/

CytoPy was authored by <a href=https://www.linkedin.com/in/burtonbiomedical/>Ross Burton</a>
and the <a href=https://www.cardiff.ac.uk/people/view/78691-eberl-matthias>Eberl Lab</a>
at <a href=https://www.cardiff.ac.uk/medicine/research/divisions/infection-and-immunity>Cardiff University Infection and Immunity Research Institute</a>

# Release notes

* 2.0.0 (stable) - This new build represents a refactored framework that is not compatible with previous builds. Expanded methods and a restructured design.
* 1.0.1 (premature) - This release corrects some major errors encountered in the flow.clustering module that was preventing clusters from being saved to the database and retrieved correctly.
* 1.0.0 (premature) - This is the first major release of CytoPy following the early release of v0.0.1 and updated in v0.0.5 and v0.1.0. This first major release includes fundamental changes to data management and therefore is not backward compatible with previous versions.

# Contributors and future directions

We are looking for open source contributors to help with the following projects:

* Graphical user interface deployed with Electron JS to expose CytoPy to scientists without training in Python
* Expansion of test coverage for version 2.0.0
* CytoPySQL: a lightweight clone of CytoPy that swaps out mongoengine for PeeWee ORM, granting the use of SQLite for those that cannot host a MongoDB service on their local machine or on Mongo Atlas
