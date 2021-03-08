****************
CytoPy Tutorials
****************

We have a number of Jupyter Notebooks that provide detailed examples of how to use CytoPy.
Broken down into the common tasks these are:

1. `Autonomous gating <https://github.com/burtonrj/CytoPyManuscript/blob/main/sup_gates.ipynb>`_
2. `Visualising and correcting batch effects <https://github.com/burtonrj/CytoPyManuscript/blob/main/01%20Validation/01%20Batch%20effects%20in%20PBMCs.ipynb>`_
3. `Supervised classification of FlowCAP <https://github.com/burtonrj/CytoPyManuscript/blob/main/01%20Validation/04%20CellClassifier%20validation%20with%20FlowCAP.ipynb>`_ and `T cell subsets <https://github.com/burtonrj/CytoPyManuscript/blob/main/01%20Validation/04%20CellClassifier%20validation%20with%20FlowCAP.ipynb>`_
4. `High dimensional clustering <https://github.com/burtonrj/CytoPyManuscript/blob/main/01%20Validation/06%20FlowSOM%20clustering%20vs%20manual%20gating.ipynb>`_
5. `Feature extraction and selection <https://github.com/burtonrj/CytoPyManuscript/blob/main/02%20Application/06%20Feature%20engineering%20and%20selection.ipynb>`_


General note on utilities
#########################

CytoPy offers a number of utility modules useful for studying cytometry data. For
the most part, the user can access the general class structures of CytoPy and not
have to touch these individual utility modules. However, you will be passing arguments
that call on these utilities, so it is useful to understand what is going on under
the hood.

**Transformations**

The *cytopy.flow.transform* module houses functionality for common data transforms applied to cytometry data:

* `Logicle (biexponential) <https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.22030">`_
* `Hyperlog <https://pubmed.ncbi.nlm.nih.gov/15700280/>`_
* `Natural log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_
* `Log (base 2) <https://numpy.org/doc/stable/reference/generated/numpy.log2.html>`_
* `Log (base 10) <https://numpy.org/doc/stable/reference/generated/numpy.log10.html>`_
* `Parametrised Log <http://flowcyt.sourceforge.net/gating/latest.pdf>`_
* `Inverse hyperbolic sine transformation <http://flowcyt.sourceforge.net/gating/latest.pdf>`_

Data transformation is handled by the *Transformer* class with a child class for each
of the logicle, hyperlog, log, and inverse hyperbolic sine transformations. A *Transformer*
is initialised with the parameters of the transformation function to be applied and
then exposes data to two functions:

* *scale* - takes a DataFrame and list of columns to transform and returns the transformed DataFrame

* *inverse_scale* - takes a DataFrame of previously transformed data and a list of columns to inverse this transformation for, and returns the DataFrame with data of the original scale

There are a couple of convenient functions in this module that construct the *Transformer* class and return the transformed data and the *Transformer*, which can then be used to inverse the transformation at a later point. These functions are:

* *apply_transform* - takes the data, transform method as a string, and the features (columns) to transform, applying the same transformation to each column.
* *apply_transform_map* - same as above except it takes a dictionary where the key is the name of the feature (column) to transform and the value is the transform method to be applied; this enables different transforms to be applied to different features.

Finally, the *transform* module also provides the *Scaler* class, a convenient wrapper for Scikit-Learn's scaling functions. The user constructs this class by specifying a *method* which then will apply a scaling function:

* "standard" - sklearn.preprocessing.StandardScaler
* "minmax" - sklearn.preprocessing.MinMaxScaler
* "robust" - sklearn.preprocessing.RobustScaler
* "maxabs" - sklearn.preprocessing.MaxAbsScaler
* "quantile" - sklearn.preprocessing.QuantileTransformer
* "yeo_johnson" - sklearn.preprocessing.PowerTransformer
* "box_cox" - sklearn.preprocessing.PowerTransformer

Making a call to this function with a DataFrame and a list of features (columns) will
apply scaling to the DataFrame. Additionally, where supported, the user can call *inverse*
on the same DataFrame and inverse the scaling performed.

**Sampling**

Throughout CytoPy when we want to take a sample of our data, often to make our operations
computationally viable, we will specify a sampling method and we will be offered the option
to pass additional keyword arguments for this sampling method. When this action occurs, we
are making a call to one of the functions in the *cytopy.flow.sampling* module. The
functions within this module are:

* *uniform_downsampling* ('uniform') - wraps the Pandas DataFrame sampling method with some additional error handling. Will downsample the given dataframe with a uniform weight given to each row.
* *faithful_downsampling* ('faithful') - an implementation of faithful downsampling as described in: Zare H, Shooshtari P, Gupta A, Brinkman R.  Data reduction for spectral clustering to analyze high throughput flow cytometry data. BMC Bioinformatics 2010;11:403
* *density_dependent_downsampling* ('density') - density dependent down-sampling to remove risk of under-sampling rare populations. Originally described in SPADE, this algorithm constructs a nearest neighbour tree to estimate local density and assigns a higher probability for sampling events in low density regions whilst also ignore outliers.

For more information regarding the parameters to control each of these methods, please
see their individual API docs.

**Dimension reduction**

Dimension reduction is a popular method for visualising cytometry data and is
useful for data exploration. CytoPy has a common function for performing
dimension reduction: *cytopy.flow.dim_reduction.dimensionality_reduction*.

This function takes the target DataFrame and a list of features (columns) to be
used when generating the desired embedding. We specify the *method* to use as a
string; supported methods should be familiar to those in the bioinformatics domains
and are: 'UMAP', 'PCA', 'KernelPCA', 'PHATE', and 'tSNE', with the objective
to expand supported methods in future versions.

By default, the DataFrame is returned with the generated embeddings as appended
columns. The column names will carry the name of the method followed by
the index of the embedding starting from 1; e.g. if we use UMAP and set
*n_components* to 4, then these columns will be "UMAP1", "UMAP2", "UMAP3" and
"UMAP4".

UMAP and PHATE are computed using the UMAP and PHATE objects from the umap and
phate libraries respectively. Other methods are provided by Scikit-Learn. The additional
parameters we might want to pass to these methods are passed as additional keyword
arguments to *dimension_reduction*. We can also specify to return the *reducer*
object and also specify to just return the embeddings and not the whole DataFrame.
