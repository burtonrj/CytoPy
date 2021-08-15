[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/pypi/pyversions/cytopy)](https://pypi.org/project/cytopy/)


# CytoPy-Widgets

This package provides a GUI wrapper for CytoPy gating, using the [ipywidgets package](https://ipywidgets.readthedocs.io/en/latest).
CytoPy-Widgets facilitates iterative batch gating of a large number of samples that should be processed in the same way.

## Prerequisites
* Please make sure you have [ipwdigets installed](https://ipywidgets.readthedocs.io/en/latest/user_install.html)
* Before using CytoPy-Widgets, you must create a new  CytoPy experiment and add samples
* When creating your experiment, please make sure that all of your control samples are added first

## Use instructions
* To start using the package, please simply create a new cell in your notebook and type (replacing `PROJECT_NAME`, `EXPERIMENT_NAME` with your actual projet and experiment names):
```
from cytopy.cytopy_widgets.cytopy_widgets import Gating_Interface
project = Project.objects(project_id=PROJECT_NAME).get()
experiment = project.get_experiment(EXPERIMENT_NAME)
Gating_Interface(experiment=experiment, source_population='root')
```

* You should see the GUI, which should look like this:
![image](https://user-images.githubusercontent.com/83017469/129480204-d16f765c-147d-4176-aadc-f3ab8fc0e538.png)
* Below this, you should see the results of the default gating method (ellipse widget) on the first sample in your experiment:
![image](https://user-images.githubusercontent.com/83017469/129480237-c261deb1-d010-4124-9367-5b6beb98c4b3.png)
* The GUI automatically shows the recommeded ellipse parameters for the first sample in your experiment, based on the automatic
fitting routines in CytoPy.
* Now you can modify the gating parameters using the GUI, and then click on `Create Gates` to preview the resulting gates (don't forget to change the target population name in the `Target pop` widget):
![image](https://user-images.githubusercontent.com/83017469/129480598-d20d1005-2827-4817-9930-c614604556d4.png)
* Keep on tweaking the ellipes parameters until you're happy with then. Once you're happy, change the `#preview` widget to `All` and then click on `Create gates again`. This will show you the results for all of your samples.
* Once you preview the results for all gates, the `Save gates` widget stop being disabled. When you click `Save gates`, the resulting populations will be saved to the CytoPy database.
* Now you can create a new cell, and start a new gating process, this time using a different source population (for example, you can use the `live cells` population that you just defined).



**Happy gating!**
