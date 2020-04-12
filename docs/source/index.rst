CytoPy - a cytometry analysis framework for Python
===================================================

In recent years there has been an explosion in Cytometry data analysis tools in the open source scientific community. This expansion is looking to soon replace traditional methods such as manual gating with sophisticated automated algorithms.

Although exciting, this can be daunting to those from a traditional immunology background and those that are new to programming. Additionally, current tools have a loose structure in the steps taken in analysis, resulting in large custom scripts, poor reproducibility, and insufficient data management.

CytoPy was created to address these issues. It was created with the general philosophy that given some cytometry data and a clinical/experimental endpoint, we wish to find what properties seperate groups (e.g. what cell populations are important for identifying a disease? What phenotypes are changing in response to a stimulus? etc). The pipeline itself is centered around a MongoDB database, is built in the Python programming language, and designed with a 'low code' API, greatly simplifying cytometry analysis.

CytoPy was authored by `Ross Burton <https://www.linkedin.com/in/burtonbiomedical/>`_ and the `Eberl Lab <https://www.cardiff.ac.uk/people/view/78691-eberl-matthias>`_ at `Cardiff University Infection and Immunity Research Institute <https://www.cardiff.ac.uk/medicine/research/divisions/infection-and-immunity>`_. CytoPy is maintained on GitHub (https://github.com/burtonrj/CytoPy) and all the latest developments can be found here. This project is a working progress and we are eager to expand and improve it's capabilities. If you would like to contribute to CytoPy please make a pull request or email us at burtonrj@cardiff.ac.uk. For news and latest developments, follow us on Twitter `@EberlLab <https://twitter.com/EberlLab>_` and `@burtondatasci <https://twitter.com/burtondatasci>`_

Our accompanying `manuscript <https://www.biorxiv.org/content/10.1101/2020.04.08.031898v2>`_ details the application of CytoPy to a novel immunophenotyping project focused on patients receiving peritoneal dialysis who were admitted on day 1 of acute peritonitis before commencing antibiotic treatment. If you have MongoDB installed (see Getting Started) and would like to replicate our findings, you can find a copy of our database `here <https://drive.google.com/drive/folders/1_Ecp-lonXa6FD6E7cTc2J9kOUQd2DRSk?usp=sharing>`_.

.. warning:: The PD database is very large (~120Gb) on account of the large quantity of single cell data contained within, please bare this in mind before downloading and rebuilding the database

To rebuild the database locally, use *mongorestore*::

	mongorestore --db CytoPy --drop /path/to/downloaded/CytoPy_database

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
