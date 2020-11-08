************************************************************
Single cell phenotype classification by supervised learning
************************************************************

There are many ways in which we can autonomously classify cells by their phenotype. CytoPy encourages the use of multiple methodologies by creating a single data repository in which the results of multiple methods can be stored and then contrasted.

One such method is supervised machine learning. Here, we label some representative data by manual or autonomous gating, train a classifier, and then predict the classification for all remaining samples. 

How do we choose representative data as training data? In the last sectuib we discuss how we can generate a "similarity matrix" for an experiment and group samples according to some statistical distance metric. We then either choose or create a sample for each group to act as training data. A classifier is trained for each group.

We can choose a reference sample using the *calculate_ref_sample* function (see previous section) or we can create a reference sample by taking a uniform sample of events from each member of our group::

	from CytoPy.data.project import Project
	from CytoPy.flow.ref import create_reference_sample
	pd_project = Project.objects(project_id='Peritonitis').get()
	exp = pd_project.load_experiment('PD_T_PDMCs')
	create_ref_sample(experiment=exp,
			  root_population='T cells',
			  samples=group_1,
			  new_file_name='group_1_training_data',
			  sample_n=1000,
			  verbose=True)

In the function call above, we pass in an instance of **Experiment** (the experiment we are currently working on). We specify a root population that is true for each biological sample and is the population we will sample vents from. We specify a list of sample IDs to sample events from, here they are contained in the variable "group_1" which corresponds to a group derived as detailed in the previous section. 

The function *create_ref_sample* doesn't return anything, instead it saves the new file to the experiment. It can then be retrieved and manipulated like any file in the experiment. We specify the file name with the argument "new_file_name". Lastly we specify how many events to sample from each biological sample.

A sample is used as training data by interpreting the populations currently associated to it. So if we take "group_1_training_data" and gate 10 populations using the **GatingStrategy** class (see `Autonomous gating`_), we can then specify 10 (or less) of those populations to be "labels" in a classification task. This is all handled by the **CellClassifier** class detailed in the next section.

Introducing the CellClassifier
===============================

The **CellClassifier** class is the base class that all supervised classifiers inherit from in CytoPy. It handles the retrieval of population data from samples, the conversion of this data into "labels" for classification, and saving predictions. The predictions of a classifier are saved as **Population**'s no different to a **Population** defined by a gate. 

You might ask, well how do classifiers handle the multi-class structure of population tree's we see in cytometry data. Take the example below:

.. image:: images/classify/tree.png

This is clearly a very complex population tree. If we wanted a classifier to identify the populations "CD3+" and "T cells", how would we do so when there are clearly overlaps? (A cell might fall inside the CD3+ gate but then not the T cell gate). **CellClassifier** supports both multi-class and multi-label prediction (however the choice of algorithm to use may be limited for multi-label prediction, see Scikit-Learn documentation for more details: https://scikit-learn.org/stable/modules/multiclass.html).

For multi-class but single label predictions (cells belong to one population and one population only), **CellClassifier** assigns a single class to each cell in the training data. For multi-label prediction, **CellClassifier** generates a dense binary matrix of shape (n_samples, n_classes). When we call subsequent methods for multi-label prediction we can specify the *threshold* a class must exceed for a positive assignment (defaults to 0.5, i.e. >50% probability of positive outcome to be assigned a class).

When the predicting new **Populations** for some unclassified **FileGroup** after training, **Populations** inherit from the 'root' population chosen when initiating the **CellClassifier**.

The target populations for prediction are given at the point of initialising a **CellClassifier** object. The user can also specify how to transform the data, whether additional scalling should be applied (e.g. min max normalisation or standard scaling) and specify how to handle issues such as class imbalance; class weights can be provided or a sampling procedure applied (see CytoPy.flow.supervised.cell_clasifier).

There are currently two **CellClassifier** classes (inheriting the behaviour of **CellClassifier**), these are:
* SklearnCellClassifier - to be used with any supervised classification model from the Scikit-Learn ecosystem (including XGBClassifier from the XGBoost library)
* KerasCellClassifier - for the construction of deep neural networks using the Keras sequential API

Creating a classifier
**********************

The **CellClassifier** object and the classes that inherit from it follow the conventions of Scikit-Learn and provides a familar API for training and prediction. Creating a classifier always follows with these steps, shown with an SklearnCellClassifier as an example.

1. We create the **CellClassifier** object::

	xgb = SklearnCellClassifier(name="xgb_classifier",
				     multi_class=False,
                             	     features=features,
                             	     target_populations=populations,
                             	     klass="XGBClassifier",
                             	     population_prefix="xgb",
                             	     params={"max_depth": 4, "subsample": 0.05})

We provide a name for the classifier, for when we save it to the database. We provide a list of features (column names) to be used for classification and we provide the labels (target populations). The *klass* argument is a string value and should correspond to a valid Scikit-Learn class or a library supported by CytoPy that follows the Scikit-Learn template (currently, beyond Scikit-Learn, we only support XGBoost). We then provide the parameters that would be used to initiate the class in the *params* argument.

2. We load in some training data using the *load_training_data* method::

	xgb.load_training_data(experiment=exp,
                       	reference="group_1_training_data",
                       	root_population="T cells")
	
3. (Optional) It is often the case with single cell data that our data suffers from 'class imbalance', that is, some populations are significantly larger than others. We can account for class imbalance by providing class weights. We can use the *auto_class_weights* method to automatically calculate some suitable class weights::

	xgb.auto_class_weights()
	
.. Note:: Some algorithms inherently do not support class weights. Make sure to research beforehand and see if your chosen algorithm does.

4. Finally, we build our model. This initiates our model and means we're ready to start training::

	xgb.build_model()
                              
Training
=========

Taking XGBoostClassifier as an example, training a model is simple, we can just call the *fit* method like you would with any Scikit-Learn model. The **CellClassifier** provides some convenience methods as well however:

* fit_train_test_split: fits the model to training data but also keeps a fraction as a 'holdout' set (size specified by *test_frac* argument). The training and holdout performance is then measured using a list of metrics (specified in the *metrics* parameter). The function returns a dictionary of training and holdout (testing) performance
* fit_cv: you can provide any cross-validator from the `Scikit-learn library<https://scikit-learn.org/stable/modules/cross_validation.html>`_ or let it default to simple Kfold cross validation. Training and testing performance across multiple folds is then returned as a list of dictionaries.

In addition to this, the **SklearnCellClassifier** class provides a few additional functions:

* hyperparameter_tuning: providing a dictionary of parameters or "parameter grid" the optimal parameters will be chosen by either grid search cross-validation or random search. See specific API for details and consult the Scikit-Learn documentation for a complete guide: https://scikit-learn.org/stable/modules/grid_search.html
* plot_learning_curve: this method will generate a learning curve using the scikit-learn utility function sklearn.model_selection.learning_curve. This can be performed either with the training data or by providing the ID of some other previously gated *FileGroup*
* plot_confusion_matrix: this will generate a new figure of confusion matrices represented by heatmaps. An example of such is shown below.

.. image:: images/classify/confusion_holdout.png

Validating
===========

When working with a new data set it is recommended that you validate the performance of your classifier by manually classifying multiple samples and assessing the performance using *validate_classifier*. This method of **CellClassifier** returns a dictionary of classification performance compared to the already existing populations. In the example below, the samples had already been classified by manual gating::

	validation_samples = ['254-05',
			      '325-01',
			      '326-01',
			      '332-01',
			      '338-01']


	val_performance = pd.DataFrame()
	for v in validation_samples:
	    result = xgb.validate_classifier(experiment=exp,
	    				      validation_id=v, 
	    				      metrics=['f1_weighted',
	    				      		'balanced_accuracy_score',
	    					     	'precision_score',
	    					     	'recall_score'],
	    				      root_population='T cells',
	    				      return_predictions=False)
	    results = pd.DataFrame(results, index=[0])
	    result['sample_id'] = v
	    val_performance = pd.concat([val_performance, result])

The dataframe "val_performance" looks like this:

.. image:: images/classify/val_performance.png

Note that metrics can be the name of any valid Scikit-Learn metric function, see Scikit-Learn documentation for details: https://scikit-learn.org/stable/modules/model_evaluation.html

The poor performance of the outlier can be investigated further by passing the feature space and labels for the validation sample to *plot_confusion_matrix*::

	x, y = xgb.load_validation(experiment=exp, validation_id='325-01', root_population='T cells')
	xgb.plot_confusion_matrix(x=x, y=y)

This produces the following confusion matrix, showing that the poor performance stems from misclassification of gamma delta T cells and unclassified events:

.. image:: images/classify/mappings.png
.. image:: images/classify/confusion_outlier.png

Predicting populations and troubleshooting with backgating
===========================================================

When we call the *predict* method, we provide the **Experiment** and the name of the sample (FileGroup) we want to predict populations for. The *predict* method will then use the model to predict the populations and return a modified **FileGroup** with the new populations assigned::

	updated_filegroup = xgb.predict(experiment=exp, 
					sample_id='325-01', 
					root_population='T cells', 
					return_predictions=False)
					
To save the results of our classifier to the **FileGroup** we would then call the *save* method on 'updated_filegroup'.

We may want to investigate further as to how the cells classified as gamma delta T cells by XGBoost compare to those classified manually. Since the *predict* method returns a modfied **FileGroup**, we can use this **CreatePlot** class and inspect the populations. A particularly useful method of this class is the *backgate* method. We can use this to directly compare the "pseudo-gate" (predictions) of the XGBoost classifier with the manual gate, by overlaying both on the parent population, the 'T cells'::

	from CytoPy.flow.plotting import CreatePlot
	plotting = CreatePlot(transform_x="logicle", transform_y="logicle")
	# We have to provide the parent population dataframe to backgate,
	# this can be retrieved from the filegroup like so...
	parent = updated_filegroup.load_population_df("T cells", transform=None)
	# Notice how we set 'transform' to None. This is because plotting will 
	# transform the data for us and we don't want to transform it twice!
	# We do the same for the populations we want to overlay on the parent
	children = {"gdt": updated_filegroup.load_population_df("gdt", transform=None),
		    "XGBoost_gdt": updated_filegroup.load_population_df("XGBoost_gdt", transform=None)}
	plotting.backgate(parent=parent,
			  children=children,
			  x="PanGD",
			  y="Vd2",
			  method={"gdt": "polygon", "XGBoost_gdt": "scatter"}

The *method* specifies how to plot the overlaid populations. We have chose to plot the manual gate as a polygon and the XGBoost generated population as a scatter plot. The above gives us the following that displays how the "poor classification" is a result of this biological sample having reduced numbers of gamma delta T cells:

.. image:: images/classify/back_gate.png


Keras
======

CytoPy extends the functionality if **CellClassifier** to deep neural networks using Keras through the **KerasCellClassifier** class. This call inherits all the functionality of **CellClassifier** but differs slightly in the way that the objects are created.

The **KerasCellClassifier** requires that an optimizer (see https://keras.io/optimizers), loss function (see https://keras.io/losses) and performance metrics (see https://keras.io/metrics) be provided when initialising the object. Additional compile kwargs can be provided with the *compile_kwargs* argument.

Layers of the neural network are defined with the **Layer** class. Individual layers should be defined with the name of the Keras class to user (*klass* argument; see https://keras.io/api/layers/) and the layer parameters in the argument *kwargs*.

Layers are then appended to a **KerasCellClassifier** layers attribute.

Keras and deep neural networks are a complex topic and we suggest further reading for a new audience. We recommend "Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow" by Aurelien Geron for further reading.

Saving classifiers
===================

Once we have defined a classifier we can save it's settings to the database for future use using the *save* method. 

.. Note:: Saving a CellClassifier to the database does not save the model, but saves the options and parameters used to create the model. When reloading the model, the user will have to call *build_model* again

For the **SklearnCellClassifier**, the underlying Scikit-Learn model can be saved to and reloaded from disk using the *save_model* and *load_model* methods, respectively.

.. Warning:: Be aware of continuity issues of saving Scikit-Learn models. Compatibility with new releases of Scikit-Learn and CytoPy are not guaranteed.


