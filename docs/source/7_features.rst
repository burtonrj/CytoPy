*************************************************
Feature extraction, selection, and summarisation
*************************************************

Once the biological samples of an experiment have been classified into phenotypically similar populations and/or clusters, we want to summarise these 'features' of the biological samples so that we can observe difference between clinical/experimental groups. CytoPy offers the feature extraction module for summarising the findings of an experiment and performing feature selection.

Summarising the proportion of cell populations/clusters
########################################################

There are multiple functions in this module for extracting and summarising the findings of an **Experiment**. The first is the *experiment_statistics* function. This takes an **Experiment** and returns a Pandas DataFrame of population statistics for each sample in the experiment::

	from CytoPy.flow.feature_extraction import experiment_statistics
	from CytoPy.data.project import Project
	pd_project = Project.objects(project_id='Peritonitis').get()
	exp = pd_project.load_experiment('PD_N_PDMCs')

	exp_stats = experiment_statistics(experiment=exp, include_subject_id=True)

The resulting dataframe will have a column for the subject ID, the sample ID (FileGroup ID), the population ID, and then statistics on the number of cells in that population and the proportion of cells compared to both the immediate parent and the root population. 

If we want to then label this dataframe with some meta-data associated to our subjects e.g. disease status, we can use the *meta_labelling* function. We provide it with the dataframe we have just created and the name of some variable stored in our Subject documents, and it creates a new column for this variable in the dataframe::

	from CytoPy.flow.feature_extraction import meta_labelling
	exp_stats = meta_labelling(experiment=exp, 
				    dataframe=exp_stats, 
				    meta_label="peritonitis")
	
The dataframe will now have a column named "peritonitis" containing a boolean value as to whether the patient had peritonitis or not.

We can generate a similar dataframe but instead look at the clustering analysis performed on a particular population. This is achieved with the `cluster_statistics` function::

	cluster_stats = cluster_statistics(experiment=exp,
					    population="T cells")

This generates a similar dataframe as before but now each row is a cluster and additional columns are included such as population ID, cluster ID, meta label, and clustering tag. The meta label and tag can be specified as arguments to this function to filter the clusters you want. Also, *population* is optional, and if it is not provided then all populations from all FileGroups are parsed for existing clusters.


Dimensionality reduction
##########################

A rapid method for detecting if there is a 'global' difference between two experimental or clinical groups is by using dimensionality reduction and plotting data points coloured according to their group. In the example below we differentiate patients with and without acute peritonitis. The dataframe 'summary' contains the proportion of cell populations identified by XGBoost and the proportion if clusters from all our experiments combined. We can use any method from CytoPy.flow.dim_reduction for the dimensionality reduction and a scatter plot is returned with data points coloured according to some label (here it is whether a patient has peritonitis or not)::
	
	from CytoPy.flow.feature_extraction import dim_reduction
	dim_reduction(summary=summary,label='peritonitis',scale=True,method='PCA')
	
.. image:: images/features/pca.png

Feature selection
###################

A simple approach for eliminating redundant variables is ranking them by their variance. The summary dataframe produced by the functions previously discussed can be passed to *sort variance* which will return a sorted dataframe for convenience.

This often isn't enough, however. If the number of features is large and we want to narrow down which are of most value to predicting some clinical or experimental endpoint, we can use L1 regularisation in a sutiable linear model to do so. L1 regularisation, also known as 'lasso' regularisation, shrinks the coefficent of less important variables to zero, producing a more sparse model. By varying the regularisation term and oberving the coefficients of all our features, we can see which features shrink more rapidly compared to others. This serves as a helpful feature selection technique, giving us the variables important for predicting some clinical or experimental endpoint.

The feature extraction module contains a function for this called *l1_feature_selection*. This function takes the feature space, a dataframe of features where each row is a different biological sample and a 'label' column specifies the label to predict. We specify which features to include in our selection and the name of the label column to predict. The model takes a search space as a tuple. This is passed to Numpy.logspace to generate a range of values to use as the different L1 regularisation terms. The first value specifies the starting value and the second the end. The search space is a *n* values (where n is the third value in this argument) between the start and end on a log scale. 

Finally we also provide the model to use. This must be a Scikit-Learn linear classifier that takes an L1 regularisation term as an argument 'C'. If None is given then a linear support vector machine is used as default::

	l1_feature_selection(feature_space=summary,
		             features=features,
		             label='peritonitis',
		             scale=True,
		             search_space=(-2, 0, 50),
		             model=None)

.. image:: images/features/l1.png


We recommend exploring the API documentation for the *feature_extraction* module. Feature selection is a large and complex topic which can be approached many ways. Some additional resources worth checking out are:

* https://scikit-learn.org/stable/modules/feature_selection.html
* https://www.coursera.org/projects/machine-learning-feature-selection-in-python
* https://academic.oup.com/bioinformatics/article/23/19/2507/185254








