CytoPy - a cytometry analysis framework for Python
===================================================

In recent years there has been an explosion in Cytometry data analysis tools in the open source scientific community. This expansion is looking to soon replace traditional methods such as manual gating with sophisticated automated algorithms.

Although exciting, this can be daunting to those from a traditional immunology background and those that are new to programming. Additionally, current tools have a loose structure in the steps taken in analysis, resulting in large custom scripts, poor reproducibility, and insufficient data management.

CytoPy was created to address these issues. It was created with the general philosophy that given some cytometry data and a clinical/experimental endpoint, we wish to find what properties seperate groups (e.g. what cell populations are important for identifying a disease? What phenotypes are changing in response to a stimulus? etc). The pipeline itself is centered around a MongoDB database, is built in the Python programming language, and designed with a 'low code' API, greatly simplifying cytometry analysis.

CytoPy is maintained on GitHub (https://github.com/burtonrj/CytoPy) and all the latest developments can be found here. CytoPy is also a working progress and we are eager to expand and improve it's capabilities. If you would like to contribute to CytoPy or have ideas regarding how it could be expanded or improved please make a pull request or email us at burtonrj@cardiff.ac.uk.

.. toctree::
    :caption: Table of Contents
    :maxdepth: 2

    Getting Started <intro>
    Data uploading <data>
    Pre-processing with autonomous gates <gating>
    Batch-effect analysis <batch>
    Supervised cell classification <classify>
    High-dimensional clustering <cluster>
    Feature extraction, selection, and description <features>
    API Reference <reference>
    License <license>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
