*************************************
Autonomous gating and pre-processing
*************************************

We were heavily inspired by the great work of OpenCyto, an autonomous gating framework developed for R Bioconductor. We recognised early on however, that autonomous gating is limited in it's capabilities. First of all, what do we mean by an "autonomous gate".

An autonomous gate is one that replicates the actions of a manual gate by applying some sort of algorithm to the same data in two dimensional space. Autonomous gates as a means of automated cytometry analysis have the following issues:

* Being a direct emulation of manual gating, they suffer from the same bias as a manually derived gating strategy
* Because the algorithm of choice is only ever applied to two-dimensional space, and therefore two variables as opposed to all available variables, the algorithm can struggle to generalise; if two biological samples deviate from one another signficantly this can result in abnormal results
* Building on the previous point, algorithms applied in this way don't take into consideration the "global topology" of the immunological landscape captured across all variables measured

This is why CytoPy focuses instead on using supervised machine learning and high-dimensional clustering that has access to all available variables when modelling the cytometry data.

Despite this, we decided to include automated gating as a function of CytoPy. The reason for this is that we found, no matter the quailty of data, some amount of 'gating' is required. Before we can classify cells autonomously we must remove debris, dead cells, and other artifacts. The efficiency and standardisation of this process can be greatly improved through the use of autonomous gates.

We actually refer to this as **semi-autonomous** gating, just for complete transparancy; some 'static' gates are often used in this task and to be honest, automated gates are not perfect and for about 10% of them some manual intervetion will be needed. We have created convenience functions to make the editing of gates easy.

Pre-processing normally follows these steps:
1. Design a gating **Template**
2. Apply that **Template** to each biological sample within an **FCSExperiment**
3. The **Template** generates **Population**s as a result of this gating process and they are saved to the underlying database in each biological samples **FCSGroup**
4. The **Template** results in an 'identical' root population being generated for each biological sample e.g. T cells or CD45+ leukocytes. This root population is the point from which supervised classification and high-dimensional clustering take place.

A **Population** is generated whenever a gate is applied or when classified by some supervised classification algorithm. High-dimensional clustering algorithms can be applied to a **Population** and the resulting clusters saved within that **Population**

The Gating and Template class
###############################

The **Gating** class is central to CytoPy's design. It is the primary class used for accessing the biological samples contained within an experiment to create and apply gates, generate **Population**s and visualise data. The **Gating** class is very powerful and we recommend checking out the API reference for details (see cytopy.gating.actions.Gating).

Often what we want to do is create a 'gating strategy': a sequence of gates applied to each biological sample, rendering the root population for each. We do this by using the **Template** class. This class directly inherits from the **Gating** class except it has the ability to save the gating sequence to the database for later use.

We initiate a **Template** object like so::

	project = Project.objects(project_id='Peritonitis').get()
	exp = project.load_experiment('PD_T_PDMCs')
	template = Template(experiment=exp, sample_id='286-03_pdmc_t', include_controls=True)

The arguments passed to **Template** are the same as those that would be passed to **Gating**. We provide **Template** with our **FCSExperiment** and the ID for the sample we want to load. For creating a gating template you should choose a biological sample that is fairly representative of all other samples in this experiment.

The *include_controls* argument specifies whether any associated control data should be loaded into the **Gating**/**Template** object. Later on we will see examples of how control data can be used.

Data is stored within the **Gating**/**Template** class as a Pandas DataFrame and can be accessed through the *data* parameter. The control data is stored as Pandas DataFrame(s) as well but is nested within a dictionary where the key is the control name.

Plotting a population
***********************

For every sample there will always be one population present by default. This is called the 'root' population. Not to be confused with what we refer to before. This 'root' is a population that contains all the events in an fcs file.

We can plot the 'root' population using the *plot_population* method. For all plotting tasks we access the embedded object **plotting**::

	template.plotting.plot_population(population_name='root', x='FSC-A', y='SSC-A', transforms={'x':None, 'y':None})

.. image:: images/gating/root.png

The transforms arugment is how we want to transform the x and y-axis. CytoPy supports all common data transformations for cytometry data (see cytopy.flow.transforms). By default a biexponential transformation is applied.


Creating and applying a gate
*****************************

We first create a gate and then we apply that gate. Before we create a gate however, we must specify the populations that gate will generate. Below we are going to create a one dimensional threshold gate that captures CD3+ and CD3- populations. This gate used the properties of a probability density function formed by gaussian KDE to find a suitable threshold that seperates regions of high density. We need to tell our gate that we expect two populations as a result. We do this using the **ChildPopulationCollection**::

	from cytopy.flow.gating.defaults import ChildPopulationCollection
	children = ChildPopulationCollection('threshold_1d')
	children.add_population('CD3+', definition='+')
	children.add_population('CD3-', definition='-')

The children object tells us:
* This is a 1D threshold gate
* It generates a population named 'CD3+' that is defined as being '+' (right of the threshold)
* It generates a population named 'CD3-' that is defined as being '-' (left of the threshold)

We can now create this gate by passing the children to the *create_gate* method::

	kwargs = dict(x='CD3', 
			transform_x='logicle', 
			kde_bw=0.05,
			peak_threshold=0.05)
	template.create_gate(gate_name='CD3_gate', 
		      parent='cells',
		      class\_='DensityThreshold',
		      method='gate_1d',
		      child_populations=children, 
		      kwargs=kwargs)

We specify the gate name, this is what we will use to refer to the gate in the future. The parent population that the gate is applied too. The type of gate we apply (class and method; see below for types of gates), the child populations produced, and the keyword arguments that are required for this gate type (again, see below for details)

Applying a gate, once created, is simple::

	template.apply('CD3_gate')

.. image:: images/gating/cd3.png


If we wanted to observe the populations currently associated to a **Gating**/**Template** object we call the *print_population_tree* method::

	template.print_population_tree()

.. image:: images/gating/tree.png

The actions described above are exactly the same for a **Gating** object. The exception is that for a **Template** object we can save the gates to our database for later use::

	template.save_new_template('Preprocessing')


The **Template** can then be reloaded to apply to further samples::
	
	template = Template(experiment=exp, sample_id='new_sample', include_controls=True)
	template.load_template('PBMCt_Preprocessing')
	template.apply_many(apply_all=True, plot_outcome=True, feedback=False)

The *apply_many* method allows you to apply many or all gates to a sample.

Types of Gates
###############

Gates fall into the following cateogores according to the type of geometric object they produce: threshold_1d, threshold_2d, cluster (polygon generated from clustering algorithm applied in two dimensions), and geom (ellipse and rectangles).

Each gate produces a **Geom** object that is saved to the **Population** and defines the 'space' in which that population is defined (e.g. the variables on the x and y axis, how they are transformed, and the coordinates in this space that "capture" the population of interest)

For every type of gate there is a class that inherits from the **Gate** class in cytopy.flow.gating.base

Each gate and their class is detailed below. Code examples are given for creating and applying a gate. Reminder: in the examples below we create gates for **Template** object, but the commands are the same for a **Gating** object.

DensityThreshold
*****************

Quantile
*********

DensityClustering
******************

MixtureModel
*************

Static
*******

Using control data
###################









