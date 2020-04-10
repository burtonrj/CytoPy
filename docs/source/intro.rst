****************
Getting started
****************

Welcome to CytoPy, a data-centric analytical framework for cytometry data. The source code for CytoPy is stored and maintained at https://github.com/burtonrj/CytoPy. You can also read our pre-print `manuscript <https://www.biorxiv.org/content/10.1101/2020.04.08.031898v2>`_. It is our hope that CytoPy opens the door to a bioinformatics approach to Cytometry analysis by using the beginner friendly programming langauge, Python. CytoPy was developed in Python 3.7. If you're new to programming that is fine, as we have added some information below for installing Python 3.7.

CytoPy makes the assumption that your hypothesis is as follows:

.. centered:: "We have collected data on humans/mice/cell-lines in X experimental/clinical conditions and we want to test for cell phenotypes that differentiate between these conditions"

We recognise that there has been an extraordinary effort to develop bioinformatics tools for addressing questions like the one above using Cytometry data. Some of these tools even feature within CytoPy itself (see https://www.biorxiv.org/content/10.1101/2020.04.08.031898v2 for details). So why CytoPy and not one of the 30+ tools in the literature? CytoPy is an agnostic framework that will allow you to apply autonomous gates, supervised classification, and high-dimensional clustering algorithms, and it achieves all this whilst providing a low-code interface and a central data repository in the form of a MongoDB database. We want to make the amazing tools in the literature more accessible to immunologists whilst improving the analysts experience.

Bioinformatics is a jungle of possible methods and an analysis tends to amount hundreds of scripts, lots of csv files, and lots of headaches. In CytoPy all experimental/clinical metadata is housed within a central database, which can be hosted locally or online. Linked to this metadata are the results of your gating, classification and clustering, and this is all stored in one central repository. Analysis is iterative and this fact has steered the design of CytoPy; an object-orientated interface built atop the `MongoEngine <http://mongoengine.org/>`_ ORM makes interacting with this database a breeze. 

Let's get started by explaining how to setup CytoPy on your local system...


Using Python
#############

For those of you who are brand-new to programming or not familiar with Python, this section will provide some helpful tips on how to get started. 

.. note:: CytoPy assumes you are familar with Python version 3, have some experience with object-orientated programming, and are happy with the concepts of Numpy arrays, Pandas DataFrames, and general data science concepts like machine learning and clustering

If the above note is daunting, please don't threat! There are lots of resources linked before and I believe with 6/8 weeks hard work anyone can grasp these concepts. So if you're brand new to Python, please start at one of the following resources:

* https://www.learnpython.org/ (Basics)
* https://www.freecodecamp.org/news/want-to-learn-python-heres-our-free-4-hour-interactive-course/ (Basics)
* https://www.youtube.com/watch?v=rfscVS0vtbw (Basics)
* https://jakevdp.github.io/PythonDataScienceHandbook/ (Data science)

To install Python 3 locally we recommend either:

* Download and install from `Python.org <https://www.python.org/downloads>`_ and then install virtualenv (see `here <https://realpython.com/python-virtual-environments-a-primer/>`_ for information about venv)
* Or, install `Anaconda <https://www.anaconda.com/>`_ which has environment management built in

We recommend making yourself familiar with programming environments before getting started. CytoPy has many dependencies and the best way to prevent any problems down the line is to keep CytoPy contained within it's own programming environment.

Installing MongoDB
###################

CytoPy assumes you have a MongoDB server up and running, either locally or in the cloud. If you wish to install MongoDB locally, then download the community edition `here <https://www.mongodb.com/download-center/community>`_. CytoPy makes the assumption that MongoDB is hosted locally, but if it is hosted remotely this can be specified when connecting to the database by providing the host address, port, and authentication data.

Although to use CytoPy you don't need to know much about MongoDB, we recommend that the user learns a bit about this powerful document-based database and how to perform simple troubleshooting and queries. To learn more we recommend checking out the resources over at FreeCodeCamp (https://www.youtube.com/watch?v=E-1xI85Zog8).

We also suggest using some sort of MongoDB GUI for troubleshooting and helping with understanding how data is stored. `Robo3T <https://robomongo.org/>`_ is a free tool that can be used for this purpose.

Installing CytoPy
##################

So you have Python 3 installed, you have MongoDB installed, and now you're ready to get started with CytoPy. First you want to create a new programming environment and activate that environment. Once inside that programming environment, either run the following command::
	
	pip3 install git+https://github.com/burtonrj/CytoPy.git

Or, alternatively, download the source code and run the setup file as so::
	
	python3 setup.py install

For a detailed overview of CytoPy we direct you to our `manuscript <https://www.biorxiv.org/content/10.1101/2020.04.08.031898v2>`_. The remaining tutorials on this site display the functionality of CytoPy by replicating the analysis described within our manuscript.




