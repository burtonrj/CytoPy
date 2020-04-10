<p align="center">
  <img src="https://github.com/burtonrj/CytoPy/blob/master/logo.png" height="25%" width="25%">
  <h1 align="center">CytoPy: a cytometry analysis framework for Python</h1>
</p>

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
simplifying cytometry analysis. We can break it all down into the following steps that can be completed within minimal 
code required:

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

For more details we refer you to our pre-print manuscript and software documentation. Our documentation contains 
a detailed tutorials for each of the above steps (https://cytopy.readthedocs.io/)

## Installation

### Python and MongoDB

CytoPy was built in Python 3.7 and uses MongoDB for data management. 

For installing MongoDB the reader should refer to https://docs.mongodb.com/manual/installation/

CytoPy assumes that the installation is local but if a remote MongoDB database is used then a host address, port and 
authentication parameters can be provided when connecting to the database, which is handled by cytopy.data.mongo_setup.

For installing Python 3 we recommend the distribution provided on <a href='https://www.python.org/downloads/'>Python.org</a> but 
 alternatively <a href='https://www.anaconda.com/'>Anaconda</a> can be used. We suggest that CytoPy be installed within an isolated 
programming environment and suggest the environment manager <a href='https://docs.python.org/3/tutorial/venv.html'>venv.</a>

### Installing CytoPy

To install CytoPy and it's requirements, first download the source code, activate the desired environment and then run the following command:

`python3 setup.py install`

Alternatively, run the following from within the desired environment:

`pip3 install git+https://github.com/burtonrj/CytoPy.git`

## License and future directions

03/Apr/2020

CytoPy is licensed under the MIT license from the Open Source Initiative. CytoPy was authored by <a href='https://www.linkedin.com/in/burtonbiomedical/'>Ross Burton</a> and the <a href='https://www.cardiff.ac.uk/people/view/78691-eberl-matthias'>Eberl Lab</a>  at <a href='https://www.cardiff.ac.uk/systems-immunity'>Cardiff University's Systems Immunity Research Institute</a>. This project is a working progress and we are eager to expand and improve its capabilities. If you would like to contribute to CytoPy please make a pull request or email us at burtonrj@cardiff.ac.uk. For news and latest developments, follow us on Twitter <a href='https://twitter.com/EberlLab'>@EberlLab</a> and <a href='https://twitter.com/burtondatasci'>@burtondatasci</a>

In future releases we are currently interested in the following:

* Incorporating data transform/normalisation procedures that mitigate or 'remove' noise as a result of batch effect. 
Methods of interest include <a href='https://arxiv.org/pdf/1610.04181.pdf'>MMD-ResNet</a>, 
<a href='https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1764-6'>BERMUDA</a>,
<a href='https://www.nature.com/articles/s41592-019-0576-7'>SAUCIE</a>, and
<a href='https://www.ncbi.nlm.nih.gov/pubmed/32151972'>scID</a>

* Improving clustering and meta-clustering by more robust methodology as described in 
    * https://www.nature.com/articles/s41598-018-21444-4
    * https://www.ncbi.nlm.nih.gov/pubmed/26492316?dopt=Abstract
    * https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-314
   
* CytoPy makes the labelling of single-cell data with meta-data simple with the integration of MongoDB and 
sophisticated object mappings. We want to therefore include the capability to model cytometry data in a direct supervised 
classification task where the objective is to predict a disease/experiment end-point and interpretation of the learned 
model reveals cell phenotypes of interest. This is demonstrated recently in the following manuscript https://www.biorxiv.org/content/10.1101/2020.02.05.934521v1 
and in 2017 in <a href='https://www.nature.com/articles/ncomms14825'>CellCNN</a>

* In the current version of CytoPy data transforms (such as logicle i.e. biexponential transform) is applied to data 
prior to plotting and is a function independent of generating plots. A more effective visualisation would be to create 
a custom Matplotlib Transform. This would also mean that the axis of plots could be displayed in decimals, making the 
plots align better visually with current practice and FlowJo outputs.



