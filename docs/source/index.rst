CytoPy - a cytometry analysis framework for Python
===================================================

In recent years there has been an explosion in cytometry data analysis tools in
the open source community This expansion is looking to soon replace traditional
methods such as manual gating with sophisticated automated algorithms.

Although exciting, this can be daunting to those unfamiliar with cytometry
bioinformatics. Current tools have a loose structure in the steps taken
in analysis, resulting in large custom scripts, poor reproducibility, and
insufficient data management. These issues become exponentially more complex
when we consider large cytometry studies spanning months or even years, and
that require integration with other data sources.

CytoPy was created to address these issues. CytoPy is designed with the following
principles:

* Common tasks are structured into modules but wherever possible, these modules are algorithm-agnostic, with the user encouraged to try different methods.
* Different methods of categorising/classifying cells (e.g. autonomous gates, high-dimensional clustering, or supervised classification) all generate the same fundamental data: Population's, which encapsulate events of similar phenotype
* The results of our analysis are stored side-by-side with meta data and other data sources in a central data repository, exposing meta data at any point in our analysis.
* Common tasks such as gating, clustering, supervised classification, plotting, and summarising your data should be achieved with as few lines of code as possible. Therefore CytoPy attempts to provide a 'low-code' API for cytometry bioinformatics.

The framework uses MongoDB as it's central data repository, chosen for it's
schema-less design and ability to scale. This means that if your study design
changes or you introduce a new data resource, CytoPy can accommodate this data
without redesigning your existing database or disrupting your workflow.

We encourage the end user to adopt MongoDB, either running on your local
machine as a service or hosted online with `Mongo Atlas <https://www.mongodb.com/cloud/atlas>`_.
Under active development is CytoPy-SQL which in the future will serve as a lighter alternative to CytoPy; this will PeeWee as it's object-relationship mapper and is suitable to use with SQLite. Although, this package will not offer the same flexibility as CytoPy.

Our accompanying `manuscript <https://www.biorxiv.org/content/10.1101/2020.04.08.031898v3>`_
details the application of CytoPy to a novel immunophenotyping project focused on
patients receiving peritoneal dialysis who were admitted on day 1 of acute peritonitis before
commencing antibiotic treatment. You can find the accompanying Jupyter Notebooks
for this study `here <https://github.com/burtonrj/CytoPyManuscript>`_.

CytoPy was authored by `Ross Burton <https://www.linkedin.com/in/burtonbiomedical/>`_
and the `Eberl Lab <https://www.cardiff.ac.uk/people/view/78691-eberl-matthias>`_
at `Cardiff University’s Systems Immunity Research Institute <https://www.cardiff.ac.uk/systems-immunity>`_.
CytoPy is maintained on GitHub (https://github.com/burtonrj/CytoPy) and all the latest developments
can be found here. We are eager to expand and improve it's capabilities and are
open to contributions and collaborations. If you would like to contribute to CytoPy
please make a pull request or email us at burtonrj@cardiff.ac.uk.
For news and latest developments, follow us on Twitter `@EberlLab <https://twitter.com/EberlLab>`_
and `@burtondatasci <https://twitter.com/burtondatasci>`_


.. toctree::
    :caption: Table of Contents
    :maxdepth: 2

    Installation <1_install>
    Introduction to the framework and managing data <2_data>
    Tutorials <3_tuts>
    API Reference <4_reference>
    License <5_license>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
