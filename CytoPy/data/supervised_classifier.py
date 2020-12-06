#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Cells can be classified by following a traditional approach of separating
data in one or two dimensional space using "gates" which can be automated
in CytoPy using the Gate and GatingStrategy classes. CytoPy also offers
an alternative approach through the CellClassifier, contained here. The
CellClassifier allows you to take gated examples, train a supervised
classification algorithm and then annotate the remaining data using
the trained model. The apparatus of CellClassifier means that the resulting
populations can be stored to a FileGroup in Populations and handled
in all other subsequent analysis.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from ..feedback import vprint, progress_bar
from ..flow.transforms import scaler
from ..flow import supervised
from ..flow import sampling
from .experiment import Experiment, FileGroup
from .population import Population, create_signature
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, KFold, learning_curve, \
    BaseCrossValidator, GridSearchCV, RandomizedSearchCV
from keras.callbacks import History
from inspect import signature
from matplotlib.axes import Axes
from warnings import warn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mongoengine
import pickle

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"

DEFAULT_METRICS = ["balanced_accuracy_score", "f1_weighted", "roc_auc_score"]


class CellClassifier(mongoengine.Document):
    """
    Supervised classification of cells using a gated example. The user
    provides an Experiment and the name of a sample which has existing
    Populations; we call this the 'reference sample'. Note, a CellClassifier
    should be defined for a single Experiment and should not be applied
    to multiple Experiments.
    These Populations are used as labels in training data, to train a
    supervised classification model. This class is the parent of two
    classes: SklearnCellClassifier and KerasCellClassifier. Please
    consult the documentation for this classes for more information.

    Attributes
    -----------
    name: str, required
        Name of the CellClassifier to save to the database. Must be
        unique
    feature: list, required
        List of markers used as input variables for classification
    multi_label: bool (default=False)
        If True, the classification problem will be treated as a
        multi-label problem; that is, a single cell can belong to
        multiple populations and the classifier will be trained to
        attribute multiple classes to a cell. This requires that the
        user provides a threshold for positivity which defaults to 0.5.
    target_populations: list, required
        List of populations to search for in the reference sample and use
        as target labels
    transform: str (optional; default="logicle")
        Transformation to apply to data prior to training/prediction
    scale: str (optional; default=None)
        Value of "standard" or "norm" should be provided if you wish
        to scale the data prior to training/prediction. Recommended for
        most methods except tree-based classifiers.
    scale_kwargs: dict
        Keyword arguments passed to scaling method, see CytoPy.flow.transform
    downsample: str (optional, default=None)
        Value of 'uniform', 'density' or 'faithful' should be provided
        if the user wishes to downsample the data prior to training.
        For downsampling methods see CytoPy.flow.sampling.
    downsample_kwargs: dict
        Keyword arguments passed to sampling method, see CytoPy.flow.sampling
    class_weights: dict (optional)
        Can be used to handle class imbalance by passing a dictionary
        of weights to associate to each population class. Alternatively
        user can use the "auto_class_weights" method to calculate balanced
        weights using the compute_class_weight function from Scikit-Learn.
    population_prefix: str (default="sml")
        Prefix added to the name of predicted populations
    verbose: bool (default=True)
        Provide feedback to stdout
    model: object
        Read-only attribute housing the classification model
    x: Pandas.DataFrame
        Training feature space
    y: Numpy.Array
        Training labels
    """
    name = mongoengine.StringField(required=True, unique=True)
    features = mongoengine.ListField(required=True)
    multi_label = mongoengine.BooleanField(default=False)
    target_populations = mongoengine.ListField(required=True)
    transform = mongoengine.StringField(default="logicle")
    scale = mongoengine.StringField(choices=["standard", "norm"])
    scale_kwargs = mongoengine.DictField()
    oversample = mongoengine.BooleanField(default=False)
    class_weights = mongoengine.DictField()
    downsample = mongoengine.StringField(choices=['uniform', 'density', 'faithful'])
    downsample_kwargs = mongoengine.DictField()
    population_prefix = mongoengine.StringField(default="sml")

    meta = {"allow_inheritance": True}

    def __init__(self, *args, **values):
        self.verbose = values.pop("verbose", True)
        self.print = vprint(self.verbose)
        self._model = None
        self.x, self.y = None, None
        super().__init__(*args, **values)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _):
        raise ValueError("Model attribute is read-only. To use a different Scikit-Learn class "
                         "define a new CellClassifier. To load a pickled model use 'load_model' method")

    def check_model_init(self):
        assert self.model is not None, "Call 'build_model' prior to fit or predict"

    def check_data_init(self):
        assert self.x is not None, "Call 'load_training_data' prior to fit"

    def load_training_data(self,
                           experiment: Experiment,
                           reference: str,
                           root_population: str):
        """
        Loads training data from the given Experiment. User must provide
        the name of an existing FileGroup associated to this Experiment
        (reference) that has the desired Populations previously created.
        The Populations should be downstream of a given "root_population",
        which acts as the starting point of analysis.

        This method populates the attributes "x" (the feature space) and
        "y" (the training labels). The training labels, y, will either be
        a 1-dimensional array, if multi_class if False, or a multi-dimensional
        array where each column is a population label, if multi_class is True.

        Parameters
        ----------
        experiment: Experiment
        reference: str
        root_population: str

        Returns
        -------
        None
        """
        self.print("Loading reference sample...")
        ref = experiment.get_sample(reference)
        self.print("Checking population labels...")
        supervised.assert_population_labels(ref=ref, expected_labels=self.target_populations)
        supervised.check_downstream_populations(ref=ref,
                                                root_population=root_population,
                                                population_labels=self.target_populations)
        self.print("Creating training data...")
        if self.multi_label:
            self.x, self.y = supervised.multilabel(ref=ref,
                                                   root_population=root_population,
                                                   population_labels=self.target_populations,
                                                   features=self.features,
                                                   transform=self.transform)
        else:
            self.x, self.y = supervised.singlelabel(ref=ref,
                                                    root_population=root_population,
                                                    population_labels=self.target_populations,
                                                    features=self.features,
                                                    transform=self.transform)
        if self.scale:
            self.print("Scaling data...")
            self.x = self._scale_data(data=self.x)
        else:
            warn("For the majority of classifiers it is recommended to scale the data (exception being tree-based "
                 "algorithms)")
        if self.downsample:
            self.print("Down-sampling...")
            self._downsample(x=self.x, y=self.y)
        if self.oversample:
            self.x, self.y = RandomOverSampler(random_state=42).fit_resample(self.x, self.y)
        self.print('Ready for training!')

    def _scale_data(self, data: pd.DataFrame):
        """
        Scales the features in the given data according to self.scale

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame
        """
        kwargs = self.scale_kwargs or {}
        data[self.features] = scaler(data,
                                     return_scaler=False,
                                     scale_method=self.scale,
                                     **kwargs)
        return data

    def auto_class_weights(self):
        """
        Compute optimal class weights using the compute_class_weights
        function from Scikit-Learn. Weights are stored as a dictionary
        in self.class_weights.

        Returns
        -------
        None
        """
        self.check_data_init()
        assert not self.multi_label, "Class weights not supported for multi-class classifiers"
        self.class_weights = supervised.auto_weights(y=self.y)

    def _downsample(self,
                    x: pd.DataFrame,
                    y: np.ndarray):
        """
        Downsample feature space (x) and corresponding labels (y)
        using the methods specified in self.downsample and
        self.downsample_kwargs

        Parameters
        ----------
        x: Pandas.DataFrame
        y: Numpy.Array

        Returns
        -------
        Pandas.DataFrame, Numpy.Array
            Sampled feature space and labels
        """
        x["y"] = y
        valid = ['uniform', 'density', 'faithful']
        kwargs = self.downsample_kwargs or {}
        assert self.downsample in valid, f"Downsample should have a value of: {valid}"
        if self.downsample == "uniform":
            if "sample_size" not in kwargs.keys():
                warn("No sample_size given, defaulting to 0.5 (will half data)")
            x = sampling.uniform_downsampling(data=x, sample_size=kwargs.get("sample_size", 0.5))
        elif self.downsample == "density":
            x = sampling.density_dependent_downsampling(data=x, **kwargs)
        else:
            if "h" not in kwargs.keys():
                warn("Parameter 'h' not given for faithful downsampling, defaulting to 0.01")
            x = sampling.faithful_downsampling(data=x, h=kwargs.get("h", 0.01))
        y = x["y"].values
        x.drop("y", inplace=True)
        return x, y

    def _fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        """
        Checks that the model has been initialised and fits the
        model to the given feature space, x, using the labels, y.
        If self.class_weights are defined, will attempt to use
        class_weights.

        NOTE: this method is a placeholder and is currently
        overwritten in both SklearnCellClassifier and KerasCellClassifier

        Parameters
        ----------
        x: Pandas.DataFrame
        y: Numpy.Array
        kwargs:
            Additional keyword arguments passed to 'fit'

        Returns
        -------
        None
        """
        self.check_model_init()
        if self.class_weights is not None:
            self.model.fit(x, y, class_weight=self.class_weights, **kwargs)
        self.model.fit(x, y, **kwargs)

    def _predict(self, x: pd.DataFrame, threshold: float):
        """
        Checks that the model has been initialised and then calls the
        predict methods of the associated model.

        NOTE: this method is a placeholder and is currently
        overwritten in both SklearnCellClassifier and KerasCellClassifier
        Parameters
        ----------
        x: Pandas.DataFrame
        threshold: float

        Returns
        -------
        Numpy.Array, Numpy.Array
        """
        self.check_model_init()
        return self.model.predict(x), self.model.predict_proba(x)

    def fit_train_test_split(self,
                             test_frac: float = 0.3,
                             metrics: list or None = None,
                             return_predictions: bool = True,
                             threshold: float = 0.5,
                             train_test_split_kwargs: dict or None = None,
                             fit_kwargs: dict or None = None):
        """
        Fit the model and test on holdout data using a test-train split
        approach. The model will be fitted to training data generated
        when "load_data" is called. If "load_data" is not previously called
        will raise an AssertionError.

        Parameters
        ----------
        test_frac: float (default=0.3)
            Proportion of training data to be kept as holdout data
            for testing
        metrics: list (optional)
            List of metrics to assess performance with. Default to:
                * Balanced accuracy
                * Weighted F1 score
                * ROC AUC score
        return_predictions: bool (default=True)
            If True, vector of predicted labels returned along with
            dictionary of classification performance
        threshold: float (default=0.5)
            If multi_class is True, this will be used to determine
            positive association to a class
        train_test_split_kwargs: dict (optional)
            Additional keyword arguments passed to call to Scikit-Learn's
            train_test_split function
        fit_kwargs: dict (optional)
            Additional keyword arguments passed to models fit method call

        Returns
        -------
        dict or (dict, Numpy.Array)
            Dictionary of training and testing performance.
            If return_predictions is True, will also return an array
            of cell class predictions.
        """
        self.check_data_init()
        self.print("==========================================")
        metrics = metrics or DEFAULT_METRICS
        train_test_split_kwargs = train_test_split_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        self.print("Spliting data into training and testing sets....")
        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=test_frac,
                                                            **train_test_split_kwargs)
        self.print("Fitting model....")
        self._fit(x=x_train, y=y_train, **fit_kwargs)
        results = dict()
        y_hat = dict()
        for key, (X, y) in zip(["train", "test"], [[x_train, y_train], [x_test, y_test]]):
            self.print(f"Evaluating {key}ing performance....")
            y_pred, y_score = self._predict(X, threshold=threshold)
            y_hat[key] = {"y_pred": y_pred, "y_score": y_score}
            results[key] = supervised.calc_metrics(metrics=metrics,
                                                   y_pred=y_pred,
                                                   y_score=y_score,
                                                   y_true=y)
        self.print("==========================================")
        if return_predictions:
            return results, y_hat
        return results

    def fit_cv(self,
               cross_validator: BaseCrossValidator or None = None,
               metrics: list or None = None,
               threshold: float = 0.5,
               split_kwargs: dict or None = None,
               fit_kwargs: dict or None = None):
        """
        Fit the model to training data using cross-validation. The model
        will be fitted to training data generated when "load_data" is
        called. If "load_data" is not previously called  will raise an
        AssertionError.

        Any Scikit-Learn cross-validator object can be used to perform
        cross validation, but if not given, will default to a basic
        K-fold cross-validation.

        Parameters
        ----------
        cross_validator: BaseCrossValidator (default=KFold)
        metrics: list (optional)
            List of metrics to assess performance with. Default to:
                    * Balanced accuracy
                    * Weighted F1 score
                    * ROC AUC score
        threshold: float (default=0.5)
            If multi_class is True, this will be used to determine
            positive association to a class
        split_kwargs: dict (optional)
            Additional keyword arguments passed to "split" call of
            cross_validator
        fit_kwargs: dict (optional)
            Additional keyword arguments passed to "fit" call of model
        Returns
        -------
        List, List
            List of dictionaries detailing training performance on each round
            List of dictionaries detailing testing performance on each round
        """
        assert not self.multi_label, "Cross-validation is not supported for multi-label classification"
        metrics = metrics or DEFAULT_METRICS
        split_kwargs = split_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        cross_validator = cross_validator or KFold(n_splits=10, random_state=42, shuffle=True)
        training_results = list()
        testing_results = list()
        for train_idx, test_idx in progress_bar(cross_validator.split(X=self.x, y=self.y, **split_kwargs),
                                                total=cross_validator.n_splits,
                                                verbose=self.verbose):
            x_train, x_test = self.x.loc[train_idx], self.x.loc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self._fit(x=x_train, y=y_train, **fit_kwargs)
            y_pred, y_score = self._predict(x=x_train,
                                            threshold=threshold)
            training_results.append(supervised.calc_metrics(metrics=metrics,
                                                            y_pred=y_pred,
                                                            y_score=y_score,
                                                            y_true=y_train))
            y_pred, y_score = self._predict(x_test, threshold=threshold)
            testing_results.append(supervised.calc_metrics(metrics=metrics,
                                                           y_pred=y_pred,
                                                           y_score=y_score,
                                                           y_true=y_test))
        return training_results, testing_results

    def fit(self, **kwargs):
        """
        Fits the model to loaded training data. If "load_data" is not
        called prior to this method, will raise AssertionError.

        Parameters
        ----------
        kwargs
            Additional keyword arguments pass to "fit"

        Returns
        -------
        None
        """
        self.check_data_init()
        self._fit(x=self.x, y=self.y, **kwargs)

    def _add_unclassified_population(self,
                                     x: pd.DataFrame,
                                     y_pred: np.ndarray,
                                     root_population: str,
                                     target: FileGroup):
        """
        For single class multi-label classification some cells will
        not be associated to any particular population (these are
        labelled as 0 when "load_data" is called). This method generates
        a Population for unlabelled cells during prediction.

        This method mutates the FileGroup object by adding a new Population.

        Parameters
        ----------
        x: Pandas.DataFrame
            Feature space
        y_pred: Numpy.Array
            Predicted labels
        root_population: str
            Starting point of analysis and parent population
        target: FileGroup
            Target FileGroup for which Populations are being predicted

        Returns
        -------
        None
        """
        idx = x.index.values[np.where(y_pred == 0)[0]]
        target.add_population(Population(population_name=f"{self.population_prefix}_Unclassified",
                                         index=idx,
                                         n=len(idx),
                                         parent=root_population,
                                         warnings=["supervised_classification"],
                                         signature=create_signature(data=x, idx=idx)))

    def predict(self,
                experiment: Experiment,
                sample_id: str,
                root_population: str,
                threshold: float = 0.5,
                return_predictions: bool = True):
        """
        Predict the population labels for cells in a FileGroup (specified
        by "sample_id") in the same Experiment as the training data (reference
        sample). Populations will be generated with the "root_population"
        as their parent. The modified FileGroup with newly predicted
        Populations will be returned. To save the Populations, call the
        "save" method of the returned FileGroup.

        Parameters
        ----------
        experiment: Experiment
            Must be the same Experiment that the training data was from
        sample_id: str
            Name of the FileGroup to predict populations for
        root_population: str
            Root of the analysis; will be the immediate parent of any
            generated Populations
        threshold: float (default=0.5)
            If multi_class is True, this will be used to determine
            positive association to a class
        return_predictions: bool (default=True)
            If True, vector of predicted labels returned along with
            the modified FileGroup object

        Returns
        -------
        FileGroup or (FileGroup, dict)
            Modified FileGroup with newly predicted Populations
            If return_predictions is True, will return a dictionary of the
            following format:
            {"y_pred": predicted labels, "y_score": confidence scores}
        """
        self.check_model_init()
        target = experiment.get_sample(sample_id)
        x = target.load_population_df(population=root_population,
                                      transform=self.transform)[self.features]
        y_pred, y_score = self._predict(x=x, threshold=threshold)
        if not self.multi_label:
            self._add_unclassified_population(x=x,
                                              y_pred=y_pred,
                                              root_population=root_population,
                                              target=target)
        for i, pop in enumerate(self.target_populations):
            if self.multi_label:
                idx = x.index.values[np.where(y_pred[:, i + 1] == 1)[0]]
            else:
                idx = x.index.values[np.where(y_pred == i + 1)[0]]
            target.add_population(Population(population_name=f"{self.population_prefix}_{pop}",
                                             index=idx,
                                             n=len(idx),
                                             parent=root_population,
                                             warnings=["supervised_classification"],
                                             signature=create_signature(data=x, idx=idx)))
        if return_predictions:
            return target, {"y_pred": y_pred, "y_score": y_score}
        return target

    def load_validation(self,
                        experiment: Experiment,
                        validation_id: str,
                        root_population: str):
        """
        Load a FileGroup from the same Experiment as the training data
        (reference_sample) and use this as validation data to test the model.
        The validation FileGroup must contain all of the Population labels
        in target_populations and these populations must be downstream
        of 'root_population'.

        Parameters
        ----------
        experiment: Experiment
        validation_id: str
        root_population: str

        Returns
        -------
        Pandas.DataFrame, Numpy.Array
            Feature space and population labels for validation FileGroup
        """
        val = experiment.get_sample(validation_id)
        assert all([x in val.tree.keys() for x in self.target_populations]), \
            f"Validation sample should contain the following populations: {self.target_populations}"
        if self.multi_label:
            x, y = supervised.multilabel(ref=val,
                                         root_population=root_population,
                                         population_labels=self.target_populations,
                                         transform=self.transform,
                                         features=self.features)
        else:
            x, y = supervised.singlelabel(ref=val,
                                          root_population=root_population,
                                          population_labels=self.target_populations,
                                          transform=self.transform,
                                          features=self.features)
        if self.scale:
            x = self._scale_data(data=x)
        return x, y

    def validate_classifier(self,
                            experiment: Experiment,
                            validation_id: str,
                            root_population: str,
                            threshold: float = 0.5,
                            metrics: list or None = None,
                            return_predictions: bool = True):
        """
        Validate the model on a FileGroup in the same Experiment as
        the training data (reference sample). The validation FileGroup
        must contain all of the Population labels in target_populations
        and these populations must be downstream of 'root_population'.
        This method returns the performance of this classifier on the
        validation sample.

        Parameters
        ----------
        experiment: Experiment
        validation_id: str
        root_population: str
        metrics: list (optional)
            List of metrics to assess performance with. Default to:
                    * Balanced accuracy
                    * Weighted F1 score
                    * ROC AUC score
        threshold: float (default=0.5)
            If multi_class is True, this will be used to determine
            positive association to a class
        return_predictions: bool (default=True)
            If True, vector of predicted labels returned along with
            dictionary of classification performance

        Returns
        -------
        dict or (dict, dict)
            Dictionary of validation performance
            If return_predictions is True, will return a dictionary of the
            following format:
            {"y_pred": predicted labels, "y_score": confidence scores}
        """
        self.check_model_init()
        metrics = metrics or DEFAULT_METRICS
        x, y = self.load_validation(experiment=experiment,
                                    validation_id=validation_id,
                                    root_population=root_population)
        y_pred, y_score = self._predict(x, threshold=threshold)
        results = supervised.calc_metrics(metrics=metrics,
                                          y_true=y,
                                          y_pred=y_pred,
                                          y_score=y_score)
        if return_predictions:
            return results, {"y_pred": y_pred, "y_score": y_score}
        return results


def _valid_multi_class(klass: str):
    """
    Checks if the specified Scikit-Learn class is valid for
    multi-class classification. If not, raises AssertionError.

    Parameters
    ----------
    klass: str

    Returns
    -------
    None
    """
    valid = ["DescisionTreeClassifier",
             "ExtraTreeClassifier",
             "ExtraTreesClassifier",
             "KNeighborsClassifier",
             "MLPClassifier",
             "RadiusNeighborsClassifier",
             "RandomForestClassifier",
             "RidgeClassifierCV"]
    err = f"Invalid Scikit-Learn class for multi-label classification, should be one of: {valid}"
    assert klass in valid, err


class SklearnCellClassifier(CellClassifier):
    """
    Supervised classification of cells using a gated example. The user
    provides an Experiment and the name of a sample which has existing
    Populations; we call this the 'reference sample'. Note, a CellClassifier
    should be defined for a single Experiment and should not be applied
    to multiple Experiments.
    These Populations are used as labels in training data, to train a
    supervised classification model. This class inherits from CellClassifier
    and provides functionality to use any Scikit-Learn classifier from
    the modules discriminant_analysis, neighbors, ensemble, or svm. This
    class also supports the use of XGBClassifier from XGBoost. If you have
    another classifier that follows the Scikit-Learn template and you would
    like to use this with CellClassifier, please raise an issue on GitHub
    or contact us at burtonrj@cardiff.ac.uk.

    Attributes
    -----------
    name: str, required
        Name of the CellClassifier to save to the database. Must be
        unique
    klass: str, required
        Name of the Scikit-Learn (or Scikit-Learn 'like') class to use
        for classification
    params: dict, optional
        Parameters used when initialising the model
    feature: list, required
        List of markers used as input variables for classification
    multi_class: bool (default=False)
        If True, the classification problem will be treated as a
        multi-class problem; that is, a single cell can belong to
        multiple populations and the classifier will be trained to
        attribute multiple classes to a cell. This requires that the
        user provides a threshold for positivity which defaults to 0.5.
    target_populations: list, required
        List of populations to search for in the reference sample and use
        as target labels
    transform: str (optional; default="logicle")
        Transformation to apply to data prior to training/prediction
    scale: str (optional; default=None)
        Value of "standard" or "norm" should be provided if you wish
        to scale the data prior to training/prediction. Recommended for
        most methods except tree-based classifiers.
    scale_kwargs: dict
        Keyword arguments passed to scaling method, see CytoPy.flow.transform
    downsample: str (optional, default=None)
        Value of 'uniform', 'density' or 'faithful' should be provided
        if the user wishes to downsample the data prior to training.
        For downsampling methods see CytoPy.flow.sampling.
    downsample_kwargs: dict
        Keyword arguments passed to sampling method, see CytoPy.flow.sampling
    class_weights: dict (optional)
        Can be used to handle class imbalance by passing a dictionary
        of weights to associate to each population class. Alternatively
        user can use the "auto_class_weights" method to calculate balanced
        weights using the compute_class_weight function from Scikit-Learn.
    population_prefix: str (default="sml")
        Prefix added to the name of predicted populations
    verbose: bool (default=True)
        Provide feedback to stdout
    model: object
        Read-only attribute housing the classification model
    x: Pandas.DataFrame
        Training feature space
    y: Numpy.Array
        Training labels
    """
    klass = mongoengine.StringField(required=True)
    params = mongoengine.DictField()

    def __init__(self, *args, **values):
        assert "klass" in values.keys(), "klass is required"
        multi_class = values.get("multi_class", False)
        if multi_class:
            _valid_multi_class(values.get("klass"))
        super().__init__(*args, **values)

    def build_model(self):
        """
        Call prior to fit or predict. Initiates model and associates
        to self.model.

        Returns
        -------
        None
        """
        params = self.params or {}
        self._model = supervised.build_sklearn_model(klass=self.klass, **params)
        if self.class_weights:
            err = "Class weights defined yet the specified model does not support this"
            assert "sample_weight" in signature(self.model.fit).parameters.keys(), err

    def _predict(self,
                 x: pd.DataFrame,
                 threshold: float = 0.5):
        """
        Overrides CellClassifier._predict. Checks that the model has
        been initialised and calls relevant predict methods. If the model
        supports prediction probabilities directly, the 'predict_proba'
        method will be called, otherwise the probability of the positive
        class is determined using the 'decision_function' method.
        If multi_class is True, then positive association to a class is
        attributed to classes with a probability that exceeds the given
        threshold.

        Parameters
        ----------
        x: Pandas.DataFrame
            Feature space
        threshold: float (default=0.5)
            Threshold of positivity for multi-class prediction

        Returns
        -------
        Numpy.Array, Numpy.Array
            Predicted labels, prediction probabilities
        """
        self.check_model_init()
        if "predict_proba" in dir(self.model):
            y_score = self.model.predict_proba(x[self.features])
        else:
            y_score = self.model.decision_function(x[self.features])
        if self.multi_label:
            y_pred = list(map(lambda yi: [int(i > threshold) for i in yi], y_score))
        else:
            y_pred = self.model.predict(x[self.features])
        return y_pred, y_score

    def _fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        """
        Fits the model to loaded training data. If "load_data" is not
        called prior to this method, will raise AssertionError.
        If class_weights are defined the signature of the chosen model
        will be inspected. If sample_weight is supported, class weights
        will be imposed, otherwise a warning will be raised.

        Parameters
        ----------
        x: Pandas.DataFrame
            Feature space
        y: Numpy.Array
            Target labels
        kwargs
            Additional keyword arguments pass to "fit"

        Returns
        -------
        None
        """
        self.check_model_init()
        if len(self.class_weights) > 0:
            if "sample_weight" in signature(self.model.fit).parameters.keys():
                sample_weight = np.array([self.class_weights.get(i) for i in y])
                self.model.fit(x, y, sample_weight=sample_weight, **kwargs)
            else:
                warn("Class weights defined yet the specified model does not support this.")
                self.model.fit(x, y, **kwargs)
        else:
            self.model.fit(x, y, **kwargs)

    def hyperparameter_tuning(self,
                              param_grid: dict,
                              method: str = "grid_search",
                              **kwargs):
        """
        Perform hyperparameter tuning using either exhaustive grid search
        or randomised grid search.

        Parameters
        ----------
        param_grid: dict
            Search space
        method: str (default="grid_search")
            Should either be "grid_search" or "random"
        kwargs:
            Keyword arguments passed to grid search method

        Returns
        -------
        GridSearchCV or RandomizedSearchCV
        """
        self.check_model_init()
        self.check_data_init()
        assert method in ["grid_search", "random"], "Method should either be 'grid_search' for " \
                                                    "exhaustive search or 'random' for randomised " \
                                                    "grid search"
        if method == "grid_search":
            search_cv = GridSearchCV(estimator=self.model,
                                     param_grid=param_grid,
                                     **kwargs)
        else:
            search_cv = RandomizedSearchCV(estimator=self.model,
                                           param_distributions=param_grid,
                                           **kwargs)
        search_cv.fit(self.x, self.y)
        return search_cv

    def plot_learning_curve(self,
                            experiment: Experiment or None = None,
                            validation_id: str or None = None,
                            root_population: str or None = None,
                            ax: Axes or None = None,
                            x_label: str = "Training examples",
                            y_label: str = "Score",
                            train_sizes: np.array or None = None,
                            **kwargs):
        """
        This method will generate a learning curve using the Scikit-Learn utility function
        sklearn.model_selection.learning_curve.
        Either use the associated training data or a validation FileGroup by providing
        the Experiment object and the ID for the validation sample (validation_id).
        This validation sample should contain the same populations as the training data,
        which must be downstream of the 'root_population'.

        Parameters
        ----------
        experiment: Experiment (optional)
            If provided, should be the same Experiment training data was derived from
        validation_id: str (optional)
            Name of the sample to use for validation
        root_population: str (optional)
            If not given, will use the same root_population as training data
        ax: Matplotlib.Axes (optional)
            Axes object to use to draw plot
        x_label: str (default="Training examples")
            X-axis labels
        y_label: str (default="Score")
            Y-axis labels
        train_sizes: Numpy.Array (optional)
            Defaults to linear range between 0.1 and 1.0, with 10 steps
        kwargs:
            Additional keyword arguments passed to sklearn.model_selection.learning_curve

        Returns
        -------
        Matplotlib.Axes
        """
        self.check_model_init()
        x, y = self.x, self.y
        if validation_id is not None:
            assert all([x is not None for x in [experiment, root_population]]), \
                "For plotting learning curve for validation, must provide validaiton ID, expeirment " \
                "object, and root population"
            x, y = self.load_validation(validation_id=validation_id,
                                        root_population=root_population,
                                        experiment=experiment)
        train_sizes = train_sizes or np.linspace(0.1, 1.0, 10)
        verbose = int(self.verbose)
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        train_sizes, train_scores, test_scores = learning_curve(self.model,
                                                                x,
                                                                y,
                                                                verbose=verbose,
                                                                return_times=False,
                                                                train_sizes=train_sizes,
                                                                **kwargs)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
        ax.legend(loc="best")
        ax.set_xlabel(xlabel=x_label)
        ax.set_ylabel(ylabel=y_label)
        return ax

    def plot_confusion_matrix(self,
                              cmap: str or None = None,
                              figsize: tuple = (10, 5),
                              x: pd.DataFrame or None = None,
                              y: np.ndarray or None = None,
                              **kwargs):
        """
        Wraps CytoPy.flow.supervised.confusion_matrix_plots (see for more details).
        Given some feature space and target labels, use the model to generate a confusion
        matrix heatmap. If x and y are not provided, will use associated training data.

        Parameters
        ----------
        cmap: str (optional)
            Colour scheme
        figsize: tuple (default=(10, 5))
            Figure size
        x: Pandas.DataFrame (optional)
            Feature space. If not given, will use associated training data. To use a validation
            dataset, use the 'load_validation' method to get relevant data.
        y: Numpy.Array (optional)
            Target labels. If not given, will use associated training data. To use a validation
            dataset, use the 'load_validation' method to get relevant data.
        kwargs:
            Additional keyword arguments passed to CytoPy.flow.supervised.confusion_matrix_plots

        Returns
        -------

        """
        assert not sum([x is not None, y is not None]) == 1, "Cannot provide x without y and vice-versa"
        if x is None:
            x, y = self.x, self.y
        assert sum([i is None for i in [x, y]]) in [0, 2], \
            "If you provide 'x' you must provide 'y' and vice versa."
        return supervised.confusion_matrix_plots(classifier=self.model,
                                                 x=x,
                                                 y=y,
                                                 class_labels=self.target_populations,
                                                 cmap=cmap,
                                                 figsize=figsize,
                                                 **kwargs)

    def save_model(self, path: str, **kwargs):
        """
        Pickle the associated model and save to disk. WARNING: be aware of continuity issues.
        Compatibility with new releases of Scikit-Learn and CytoPy are not guaranteed.

        Parameters
        ----------
        path: str
            Where to save on disk
        kwargs:
            Additional keyword arguments passed to pickle.dump call

        Returns
        -------
        None
        """
        pickle.dump(self.model, open(path, "wb"), **kwargs)

    def load_model(self, path: str, **kwargs):
        """
        Load a pickled model from disk. WARNING: be aware of continuity issues.
        Compatibility with new releases of Scikit-Learn and CytoPy are not guaranteed.
        The loaded model must correspond to the expected method for this CellClassifier.

        Parameters
        ----------
        path: str
            Where to save on disk
        kwargs:
            Additional keyword arguments passed to pickle.dump call

        Returns
        -------
        None
        """
        model = pickle.load(open(path, "rb"), **kwargs)
        assert self.klass in str(type(model)), \
            f"Loaded model does not match Classifier, expected type: {self.klass}"
        self._model = model


class Layer(mongoengine.EmbeddedDocument):
    """
    Neural networks consist of layers and in the Keras framework these can be represented
    by individual objects. The Layer document is embedded in the KerasCellClassifier
    to define the types of layers to use when initialising and compiling the Keras model.

    Attributes
    -----------
    klass: str
        Name of the keras class to use; see Keras layers API for valid class names
    kwargs: dict
        Parmeters to use when initialising Keras layer
    """
    klass = mongoengine.StringField()
    kwargs = mongoengine.DictField()


class KerasCellClassifier(CellClassifier):
    """
    Supervised classification of cells using a gated example. The user
    provides an Experiment and the name of a sample which has existing
    Populations; we call this the 'reference sample'. Note, a CellClassifier
    should be defined for a single Experiment and should not be applied
    to multiple Experiments.
    These Populations are used as labels in training data, to train a
    supervised classification model. This class inherits from CellClassifier
    and provides functionality to use deep neural networks implemented with
    Keras. CytoPy uses the Keras sequential API and layers are defined and
    stored as Layer documents embedded within this class. Specify the name
    of a valid Keras layer, along with initialisation parameters, and each
    time the KerasCellClassifier is initialised, the Keras model will be
    created and compiled. Layers should be added in the order in which
    they should occur in the neural network.

    Attributes
    -----------
    name: str, required
        Name of the CellClassifier to save to the database. Must be
        unique
    layers: list
        List of Layer objects
    optimizer: str
        Optimisation function to use (see https://keras.io/optimizers)
    loss: str
        Loss function (see https://keras.io/losses)
    metrics: list
        List of metrics to use to measure performance (see https://keras.io/metrics)
    compile_kwargs: dict
        Additional keyword arguments to pass when 'compile' method is called
    feature: list, required
        List of markers used as input variables for classification
    multi_class: bool (default=False)
        If True, the classification problem will be treated as a
        multi-class problem; that is, a single cell can belong to
        multiple populations and the classifier will be trained to
        attribute multiple classes to a cell. This requires that the
        user provides a threshold for positivity which defaults to 0.5.
    target_populations: list, required
        List of populations to search for in the reference sample and use
        as target labels
    transform: str (optional; default="logicle")
        Transformation to apply to data prior to training/prediction
    scale: str (optional; default=None)
        Value of "standard" or "norm" should be provided if you wish
        to scale the data prior to training/prediction. Recommended for
        most methods except tree-based classifiers.
    scale_kwargs: dict
        Keyword arguments passed to scaling method, see CytoPy.flow.transform
    downsample: str (optional, default=None)
        Value of 'uniform', 'density' or 'faithful' should be provided
        if the user wishes to downsample the data prior to training.
        For downsampling methods see CytoPy.flow.sampling.
    downsample_kwargs: dict
        Keyword arguments passed to sampling method, see CytoPy.flow.sampling
    class_weights: dict (optional)
        Can be used to handle class imbalance by passing a dictionary
        of weights to associate to each population class. Alternatively
        user can use the "auto_class_weights" method to calculate balanced
        weights using the compute_class_weight function from Scikit-Learn.
    population_prefix: str (default="sml")
        Prefix added to the name of predicted populations
    verbose: bool (default=True)
        Provide feedback to stdout
    model: object
        Read-only attribute housing the classification model
    x: Pandas.DataFrame
        Training feature space
    y: Numpy.Array
        Training labels
    """
    layers = mongoengine.EmbeddedDocumentListField(Layer)
    optimizer = mongoengine.StringField()
    loss = mongoengine.StringField()
    metrics = mongoengine.ListField()
    compile_kwargs = mongoengine.DictField()

    def build_model(self):
        """
        Build and compile Keras model. Model is then associated to self.model.

        Returns
        -------
        None
        """
        kwargs = self.compile_kwargs or {}
        self._model = supervised.build_keras_model(layers=self.layers,
                                                   optimizer=self.optimizer,
                                                   loss=self.loss,
                                                   metrics=self.metrics,
                                                   **kwargs)

    def _predict(self,
                 x: pd.DataFrame,
                 threshold: float = 0.5):
        """
        Overrides parent _predict method to facilitate Keras predict methods. If multi_class is True,
        then threshold is used to assign labels using the predicted probabilities; positive association
        where probability exceeds threshold.

        Parameters
        ----------
        x: Pandas.DataFrame
            Feature space
        threshold: float (default=0.5)
            Threshold for positivity when multi_class is True

        Returns
        -------
        Numpy.Array, Numpy.Array
            Predicted labels, prediction probabilities
        """
        self.check_model_init()
        y_score = self.model.predict(x)
        if self.multi_label:
            y_pred = list(map(lambda yi: [int(i > threshold) for i in yi], y_score))
        else:
            y_pred = self.model.predict_classes(x)
        return y_pred, y_score

    def _fit(self,
             x: pd.DataFrame,
             y: np.ndarray,
             epochs: int = 100,
             validation_x: pd.DataFrame or None = None,
             validation_y: np.ndarray or None = None,
             **kwargs):
        """
        Overwrites the _fit method of CellClassifier to support Keras classifier.
        If a validation feature space and labels are provided, then these are passed
        to 'validation_data' of keras 'fit' method.

        Parameters
        ----------
        x: Pandas.DataFrame
            Training feature space to fit
        y: Numpy.Array
            Training labels
        epochs: int (default=100)
            Number of training rounds
        validation_x: Pandas.DataFrame
            Validation feature space
        validation_y: Numpy.Array
            Validation labels
        kwargs:
            Additional keyword arguments passed to fit method of Keras sequential API

        Returns
        -------
        Keras.callbacks.History
            Keras History object
        """
        if validation_x is not None:
            assert validation_y is not None, "validation_y cannot be None if validation_x given"
            return self.model.fit(x, y, epochs=epochs, validation_data=(validation_x, validation_y), **kwargs)
        return self.model.fit(x, y, epochs=epochs, **kwargs)

    def fit(self,
            validation_frac: float or None = 0.3,
            train_test_split_kwargs: dict or None = None,
            epochs: int = 100,
            **kwargs):
        """
        Fit the Keras model to the associated training data. If 'validation_frac' is provided,
        then a given proportion of the training data will be set apart and given to
        the 'validation_data' parameter to Keras fit method of the sequential API.
        The validation data is created using the train_test_split function from
        Scikit-Learn; additional keyword arguments can be provided as a dictionary
        with 'train_test_split_kwargs'.

        Parameters
        ----------
        validation_frac: float (optional; default=0.3)
            Proportion of training data to set aside for validation
        train_test_split_kwargs: dict (optional)
            Additional keyword arguments for train_test_split function from Scikit-Learn
        epochs: int (default=100)
            Number of training rounds
        kwargs:
            Additional keyword arguments passed to fit method of keras sequential API
        Returns
        -------
        Keras.callbacks.History
            Keras History object
        """
        train_test_split_kwargs = train_test_split_kwargs or {}
        validation_x, validation_y = None, None
        x, y = self.x, self.y
        if validation_frac is not None:
            x, validation_x, y, validation_y = train_test_split(self.x,
                                                                self.y,
                                                                test_size=validation_frac,
                                                                **train_test_split_kwargs)
        self.check_data_init()
        return self._fit(x=x, y=y, validation_x=validation_x, validation_y=validation_y, epochs=epochs, **kwargs)

    def plot_learning_curve(self,
                            history: History or None = None,
                            ax: Axes or None = None,
                            figsize: tuple = (10, 10),
                            plot_kwargs: dict or None = None,
                            **fit_kwargs):
        """
        This method will generate a learning curve using the History object generated
        from the fit method from the Keras sequential API.

        Parameters
        ----------
        history: History (optional)
            If not given, then the 'fit' method will be called and use the associated
            training data.
        ax: Matplotlib.Axes
        figsize: tuple (default=(10,10))
        plot_kwargs: dict (optional)
            Keyword arguments passed to Pandas.DataFrame.plot method
        fit_kwargs:
            Keyword arguments passed to fit method if 'history' is not given

        Returns
        -------
        Matplotlib.Axes
        """
        history = history or self.fit(**fit_kwargs)
        plot_kwargs = plot_kwargs or {}
        ax = ax or plt.subplots(figsize=figsize)[1]
        return pd.DataFrame(history.history).plot(ax=ax, **plot_kwargs)
