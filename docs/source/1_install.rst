****************
Getting started
****************

.. note:: CytoPy assumes you are familiar with Python version 3,
have some experience with object-orientated programming, Numpy and Pandas.
Preferably you also have some experience with Scikit-Learn and general
data science principles.

If the above note is daunting, please don't threat! There are lots of resources
linked below and I believe with 6/8 weeks hard work anyone can grasp enough to
start using CytoPy. So if you're brand new to Python, please start at one of the
following resources:

* https://www.learnpython.org/ (Basics)
* https://www.freecodecamp.org/news/want-to-learn-python-heres-our-free-4-hour-interactive-course/ (Basics)
* https://www.youtube.com/watch?v=rfscVS0vtbw (Basics)
* https://jakevdp.github.io/PythonDataScienceHandbook/ (Data science)

Docker
#######

If you're familiar with Docker and would like to avoid installing CytoPy and
MongoDB on your local system, you can obtains a docker image for CytoPy
`here <>`_.

.. note:: CytoPy is cross-platform compatible but has been tested on Windows 10,
Ubuntu 20.04, and Ubuntu 18.04. If you experience issues on other platforms please
raise an issue on GitHub.

Installing MongoDB
###################

Regardless of your operating system, you must `install MongoDB on your local machine
<https://docs.mongodb.com/manual/administration/install-community/>`_. Alternatively,
you can host your MongoDB database on Mongo Atlas. Connecting to a database
on Mongo Atlas from CytoPy will be discussed in the next section.


Installing CytoPy
##################

**Step 1: install Python**

We recommend installing `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_, this
is a popular data science platform that bundles Python, R, and very intuitive environment
manager together. This also comes with a handy graphical user interface for managing
all your software.

Alternatively, you can download and install from `Python.org <https://www.python.org/downloads>`_.
For this approach we recommend installing virtualenv (see `here <https://realpython.com/python-virtual-environments-a-primer/>`_
for information about venv) to manage your Python environments.

**Step 2: setup an environment**

1. If you're on a Windows machine, then open Anaconda Prompt, either from the start menu
or from the Anaconda Navigator. On Linux or Max, open a new terminal.
2. Create an environment with the following command::

    conda create --name CytoPy python=3.8

When prompted say "yes". This will create an isolated programming environment where we
will install CytoPy. When we want to work in this environment we use the following
command::

    conda activate CytoPy

**Step 3: install CytoPy**

3. Inside our environment, we want to call the following in this order::

    pip install numpy
    pip install cytopy

.. warning::
    It is vital that we install Numpy prior to installing CytoPy. This is because
    FlowUtils, a vital dependency of CytoPy, requires Numpy to compile necessary
    C extensions. It is recommended to use pip and not conda to install Numpy
    and CytoPy.

4. Next we recommend you install Jupyter and IPython to interact with CytoPy. You
can also use CytoPy in the Python console or using Spyder. To install Jupyter and
IPython you can run::

    conda install jupyter ipython

To make your CytoPy environment available to Jupyter, you want to run the following::

    python -m ipykernel install --user --name=CytoPy

Now when you launch Jupyter Notebooks or Jupyter Lab, our CytoPy environment
will be an available kernel.
