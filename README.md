<p align="center">
  <img src="https://github.com/burtonrj/CytoPy/blob/master/logo.png" height="25%" width="25%">
  <h1 align="center">CytoPy: a cytometry analysis framework for Python</h1>
</p>

<b>CytoPy is under peer-review and active development. A stable release is scheduled for 2021 but in the meantime some functionality may change</b>

In recent years there has been an explosion in Cytometry data analysis tools in the open source scientific community.
This expansion is looking to soon replace traditional methods such as manual gating with sophisticated automated algorithms.

Although exciting, this can be daunting to those from a traditional immunology background and those 
that are new to programming. Additionally, current tools have a loose structure in the steps taken in analysis, 
resulting in large custom scripts, poor reproducibility, and insufficient data management.

CytoPy was created to address these issues. It was created with the general philosophy that given some 
cytometry data and a clinical/experimental endpoint, we wish to find what properties separate groups (e.g. what cell populations
are important for identifying a disease? What phenotypes are changing in response to a stimulus? etc). 
The pipeline itself is centered around a MongoDB database, is built in  the Python programming language, 
and designed with a 'low code' API, greatly 
simplifying cytometry analysis. We can break it down into the following steps that can be completed with minimal 
code required:

1. Data uploading
2. Pre-processing
3. Quantifying inter-sample variation and choosing training data
4. Supervised cell classification 
5. High-dimensional clustering
6. Feature extraction, selection, and description

You will notice that we perform both supervised cell classification and high-dimensional clustering.
Supervised classification being training samples, gated according to some 'gating strategy', being used
to train a classifier. Alternatively high-dimensional clustering (by PhenoGraph or FlowSOM) involves clustering 
cells in a completely unbiased fashion. CytoPy provides access to both methodologies as we observe 
that both have benefits and failings.

CytoPy is algorithm agnostic and provides a general interface for accessing the tools provided by Scikit-Learn whilst following the terminology and signatures common to this library. If you would like to expand CytoPy and add additional methods for autonomous gating, supervised classificaiton or high dimensional clustering, please contact me at burtonrj@cardiff.ac.uk, raise an issue or make a pull request.

For more details we refer you to our pre-print <a href='https://www.biorxiv.org/content/10.1101/2020.04.08.031898v2'>manuscript</a> and software documentation. Our documentation contains 
a detailed tutorials for each of the above steps (https://cytopy.readthedocs.io/)

## Installation

### Python and MongoDB

CytoPy was built in Python 3.7 and uses MongoDB for data management. 

For installing MongoDB the reader should refer to https://docs.mongodb.com/manual/installation/

CytoPy assumes that the installation is local but if a remote MongoDB database is used then a host address, port and 
authentication parameters can be provided when connecting to the database, which is handled by cytopy.data.mongo_setup.

For installing Python 3 we recommend <a href='https://www.anaconda.com/'>Anaconda</a>, which also provides a convenient environment manager <a href='https://docs.python.org/3/tutorial/venv.html'>. We suggest that you always keep CytoPy contained within its own programming environment.</a>

### Installing CytoPy

First, due to CytoPy's dependencies Cython and Numpy must be installed. This can be achieved using the command:

`pip3 install cython`

`pip3 install numpy`

To install CytoPy and it's requirements, first download the source code, activate the desired environment and then run the following command:

`python3 setup.py install`

## License

03/Apr/2020

CytoPy is licensed under the MIT license from the Open Source Initiative. CytoPy was authored by <a href='https://www.linkedin.com/in/burtonbiomedical/'>Ross Burton</a> and the <a href='https://www.cardiff.ac.uk/people/view/78691-eberl-matthias'>Eberl Lab</a>  at <a href='https://www.cardiff.ac.uk/systems-immunity'>Cardiff University's Systems Immunity Research Institute</a>. This project is a working progress and we are eager to expand and improve its capabilities. If you would like to contribute to CytoPy please make a pull request or email us at burtonrj@cardiff.ac.uk. For news and latest developments, follow us on Twitter <a href='https://twitter.com/EberlLab'>@EberlLab</a> and <a href='https://twitter.com/burtondatasci'>@burtondatasci</a>



