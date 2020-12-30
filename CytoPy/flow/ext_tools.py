from ..feedback import progress_bar
from .descriptives import box_swarm_plot
from .transform import apply_transform
from warnings import warn
from lmfit.models import LinearModel, QuadraticModel, PolynomialModel
from lmfit import Model
from abc import ABC
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin
import pandas as pd
import numpy as np

INVERSE_LOG = {"log": lambda x: np.e ** x,
               "log2": lambda x: 2 ** x,
               "log10": lambda x: 10 ** x}


def four_param_logit(x: np.ndarray,
                     min_: float,
                     max_: float,
                     inflection_point: float,
                     coef: float):
    """
    Four parameter logistic function.

    Parameters
    ----------
    x: Numpy.Array
    min_: float
        The minimum asymptote i.e. where the response value should be 0 for the standard concentration
    max_: float
        The maximum asymptote i.e. where the response value where the standard concentration is infinite
    inflection_point: float
        Where the curvature changes direction or sign
    coef: float
        The coefficient or steepness of the slope

    Returns
    -------
    Numpy.Array
    """
    n = (min_ - max_)
    d = (1 + (x / inflection_point) ** coef)
    return max_ + (n / d)


def five_param_logit(x: np.ndarray,
                     min_: float,
                     max_: float,
                     inflection_point: float,
                     coef: float,
                     asymmetry_factor: float = 1.0):
    """
    Five parameter logistic function.

    Parameters
    ----------
    x: Numpy.Array
    min_: float
        The minimum asymptote i.e. where the response value should be 0 for the standard concentration
    max_: float
        The maximum asymptote i.e. where the response value where the standard concentration is infinite
    inflection_point: float
        Where the curvature changes direction or sign
    coef: float
        The coefficient or steepness of the slope
    asymmetry_factor: float
        The asymmetry factor; where asymmetry_factor = 1, we have a symmetrical curve around the inflection
        point and so we have a four parameter logistic equation

    Returns
    -------
    Numpy.Array
    """
    n = (min_ - max_)
    d = (1 + (x / inflection_point) ** coef) ** asymmetry_factor
    return max_ + (n / d)


class Logit(Model, ABC):
    """
    Logistic curve fitting model with either four or five parameters. Inherits from lmfit.Model.
    When constructed user should specify whether to use four or five parameter fit by specifying
    True or False for the 'five_parameter_fit' parameter. If 'use_default_params' is set to True, then
    the following parameter hints will be set (name: value [lower bound, upper bound]):
    * min_: 0 [-1e4,  1e4]
    * max_: 1e4 [1, 1e6]
    * inflection_point: 0.5 [1e-4, 1]
    * coef: 1 [1e-4, 1e4]
    * asymmetry_factor (only for five parameter fit): 0.5 [1e-4, 1]

    If you feel that these default starting parameters do not reflect your dataset, then set "use_default_params"
    to False and set parameter hints as per lmfit docs:
    https://lmfit.github.io/lmfit-py/model.html#initializing-values-with-model-make-params
    """

    def __init__(self,
                 five_parameter_fit: bool = True,
                 use_default_params: bool = True,
                 **kws):
        func = five_param_logit
        if not five_parameter_fit:
            func = four_param_logit
        super().__init__(func, **kws)
        if use_default_params:
            self.set_param_hint(name="min_",
                                value=0,
                                min=-1e4,
                                max=1e4)
            self.set_param_hint(name="max_",
                                value=1e4,
                                min=1,
                                max=1e6)
            self.set_param_hint(name="inflection_point",
                                value=0.5,
                                min=1e-4,
                                max=1)
            self.set_param_hint(name="coef",
                                value=1,
                                min=1e-4,
                                max=1e4)
            if five_parameter_fit:
                self.set_param_hint(name="asymmetry_factor",
                                    value=0.5,
                                    min=1e-4,
                                    max=1)


def default_models(model: str,
                   model_init_kwargs: dict or None = None):
    """
    Generates a default Model object of either Linear, Polynomial, Quadratic or Logistic function.

    Parameters
    ----------
    model: str
        Should be one of linear', 'quad', 'poly', or 'logit'
    model_init_kwargs: dict, optional
        Additional keyword arguments passed when constructing Model

    Returns
    -------
    Model

    Raises
    -------
    ValueError
        If model is not one of 'linear', 'quad', 'poly', or 'logit'
    """
    model_init_kwargs = model_init_kwargs or {}
    if model == "linear":
        model = LinearModel(**model_init_kwargs)
        model.set_param_hint(name="intercept", value=1, min=-1e9, max=1e9)
        model.set_param_hint(name="slope", value=0.5, min=1e-9, max=1e9)
        return model
    if model == "quad":
        model = QuadraticModel(**model_init_kwargs)
        model.set_param_hint("a", value=0.5, min=1e-5, max=1e5)
        model.set_param_hint("b", value=0.5, min=1e-5, max=1e5)
        model.set_param_hint("c", value=0.5, min=1e-5, max=1e5)
        return model
    if model == "poly":
        degree = model_init_kwargs.pop("degree", 2)
        model = PolynomialModel(degree=degree, **model_init_kwargs)
        for i in range(degree):
            model.set_param_hint(name=f"c{i}", value=0.5, min=1e-9, max=1e9)
        return model
    if model == "logit":
        return Logit(use_default_params=True, **model_init_kwargs)
    raise ValueError("Invalid model, must be one of: 'linear', 'quad', 'poly', or 'logit'")


def assert_fitted(method):
    """
    Given a function method of AssayTools, inspect the self argument to access
    standard curves and check that the curve has been fitted. Raises assertion error if
    condition is not met.

    Parameters
    ----------
    method: callable

    Returns
    -------
    callable
        Original method

    Raises
    ------
    AssertionError
    """

    def wrapper(*args, **kwargs):
        assert len(args[0].standard_curves) != 0, "Standard curves have not been computed; call 'fit' prior to " \
                                                  "additional functions"
        if "analyte" in kwargs.keys():
            assert kwargs.get("analyte") in args[0].standard_curves.keys(), \
                f"Standard curve not detected for {kwargs.get('analyte')}; call 'fit' prior to additional functions"
        return method(*args, **kwargs)

    return wrapper


def subtract_background(data: pd.DataFrame,
                        background_id: str,
                        analytes: list):
    """
    Given a DataFrame of assay data (as described in the documentation for AssayTools), subtract
    background using the rows with a Sample ID corresponding to the 'background_id'.

    Parameters
    ----------
    data: Pandas.DataFrame
    background_id: str
    analytes: list
        List of columns corresponding to analytes

    Returns
    -------
    Pandas.DataFrame
    """
    data = data.copy()
    assert background_id in data["Sample"].values, "Invalid background_id, not found in Sample column"
    background = data[data.Sample == background_id][analytes].copy()
    data = data[~data.index.isin(background.index)].copy()
    for analyte, mean_ in background.mean().to_dict().items():
        data[analyte] = data[analyte] - mean_
        data[analyte] = data[analyte].apply(lambda x: x if x > 0 else 0.00000001)
    return data


class AssayTools:
    """
    Tools for analysis of plate based assay data such as ELISAs and Luminex assays.
    Calculate standard curves, predict concentrations, transform data, as well as
    access to various plotting functions and statistical tests.

    AssayTools makes heavy use of the lmfit library for fitting curves to data. We recommend the user
    consults their documentation for more information and troubleshooting: https://lmfit.github.io/

    The results of fitted curves are stored in the attribute 'predictions' as a dictionary where each key
    is the analyte name and the value a nested dictionary containing the transform applied and the ModelResult
    object.

    Attributes
    ----------
    raw: Pandas.DataFrame
        Raw unaltered assay data. Will also contain any associated meta data if the subject ID is
        provided.
    predictions: Pandas.DataFrame
        Predicted concentrations of analytes using standard curves
    analytes: list
        List of analytes being studied
    concentrations: Pandas.DataFrame
        Concentrations corresponding to each standard control
    standard_curves: dict
        Fitted functions for each analyte
    """

    def __init__(self,
                 data: pd.DataFrame,
                 conc: pd.DataFrame,
                 standards: list,
                 background_id: str or None = None,
                 analytes: list or None = None):
        """
        Parameters
        ----------
        data: Pandas.DataFrame
            Assay data. Must contain the column "Sample" with each value being an individual biological
            sample; include duplicates as individual rows. Standards should be included as samples, so
            with each standard in its own row; again, including duplicates. Additional columns should
            correspond to analytes. The user can also include an additional row called "subject_id"
            for matching rows to meta data.
        conc: Pandas.DataFrame
            Concentrations of each analyte for each standard control. There should be one column named "analyte"
            with a unique row for each analyte. The remaining columns should correspond to the standard controls.
            Standards will be interpreted in the order given in the "standards" parameter.
        standards: list
            List of unique standard names. Order matters! Standards will be treated in this order as highest
            concentration to lowest.
        background_id: str, optional
            Sample ID corresponding to rows to be used for subtracting background.
        analytes: list, optional
            List of analytes measured. If not given, all columns in data not equal to "Sample" or "subject_id"
            will be treated as analytes.
        """

        self.analytes = analytes or [x for x in data.columns if x not in ["Sample", "subject_id"]]
        assert "Sample" in data.columns, "Invalid DataFrame missing 'Sample' column"
        assert all([x in data.columns for x in self.analytes]), "One or more listed analytes missing from data"
        assert all([x in data["Sample"].values for x in standards]), \
            "One or more listed standards missing from Sample column"
        assert "analyte" in conc.columns, "Analyte column missing from concentrations dataframe"
        assert all([x in conc.columns.values for x in standards]), \
            "One or more of the specified standards missing from concentrations dataframe"
        assert all([x in conc["analyte"].values for x in self.analytes]), \
            "One or more of the specified analytes missing from concentrations dataframe"
        self.concentrations = conc
        self.standard_curves = dict()
        self._predictions = dict()
        self.standards = standards
        self.raw = data
        if background_id:
            self.raw = subtract_background(data=data,
                                           background_id=background_id,
                                           analytes=self.analytes)

    @property
    def predictions(self):
        x = pd.DataFrame(self._predictions)
        other_vars = [c for c in self.raw.columns if c not in self.analytes]
        for c in other_vars:
            x[c] = self.raw[~self.raw.Sample.isin(self.standards)][c].values
        return x

    @predictions.setter
    def predictions(self, _):
        raise ValueError("Predictions is a read-only property. Call fit_predict to fit standard curves and "
                         "popualte predictions.")

    @property
    def predictions_linear(self):
        x = self.predictions
        for analyte in self._predictions.keys():
            transform = self.standard_curves.get(analyte).get("transform")
            if transform in ["log", "log2", "log10"]:
                x[analyte] = x[analyte].apply(INVERSE_LOG.get(transform))
            elif transform is not None:
                warn(f"Transform {transform} applied to analyte {analyte} does not have a supported inverse function")
        return x

    def _prepare_standards_data(self,
                                analyte: str,
                                transform: str or None = None):
        """
        Prepare the standard concentration data for a given analyte using the raw data.

        Parameters
        ----------
        analyte: str
        transform: str, optional

        Returns
        -------
        Pandas.DataFrame
        """
        standards = self.raw[self.raw.Sample.isin(self.standards)][["Sample", analyte]].copy()
        standard_concs = self.concentrations[self.concentrations.analyte == analyte].copy()
        standard_concs = standard_concs[self.standards].melt(var_name="Sample", value_name="conc")
        standards = standards.merge(standard_concs, on="Sample")
        if transform:
            standards = apply_transform(standards,
                                        transform_method=transform,
                                        features_to_transform=[analyte, "conc"])
        return standards

    def _fit(self,
             model: Model,
             transform: str or None,
             analyte: str,
             params: dict or None = None,
             guess_start_params: bool = True,
             **kwargs):
        """
        Fit the standard curve function for a single analyte.

        Parameters
        ----------
        model: Model
        transform: str
        analyte: str
        params: dict, optional
            Optional starting parameters; will overwrite defaults
        kwargs:
            Additional keyword arguments to pass to Model.fit call

        Returns
        -------
        None
        """
        params = params or {}
        standards = self._prepare_standards_data(analyte=analyte, transform=transform)
        if guess_start_params:
            try:
                params = model.guess(data=standards["conc"].values,
                                     x=standards[analyte].values)
            except NotImplementedError:
                params = model.make_params(**params)
        else:
            params = model.make_params(**params)
        self.standard_curves[analyte] = {"transform": transform,
                                         "model_result": model.fit(standards["conc"].values,
                                                                   params=params,
                                                                   x=standards[analyte].values,
                                                                   **kwargs)}

    def fit(self,
            model: Model or str,
            transform: str or None = None,
            analyte: str or None = None,
            starting_params: dict or None = None,
            guess_start_params: bool = True,
            model_init_kwargs: dict or None = None,
            **kwargs):
        """
        Fit a function to generate one or more standard curves. The standard curves
        are generated using the lmfit library (https://lmfit.github.io/), which uses least squares regression.
        A Model object should be provided or a string value which will load a default model for convenience.
        If starting_params is provided, then the specified starting parameters will be used for the initial fit,
        otherwise defaults are used.
        The resulting fit generates a ModelResult object which is stored in the standard_curves attribute, which is a
        dictionary where the key corresponds to the analyte and the value a nested dictionary like so:

        {"transform": transformation applied prior to fitting,
         "model_result": ModelResult object}

         For more details regarding a ModelResult object, see the lmfit documentation here:
         See https://lmfit.github.io/lmfit-py/model.html?highlight=modelresult#the-modelresult-class

        If the function fails to estimate the optimal parameters, indicated by a RuntimeError,
        you can try modifying the starting parameters or increasing the number of
        evaluations by passing a value for max_nfev (usually 1000-10000 will work) in kwargs.

        Parameters
        ----------
        model: Model or str
            A valid lmfit.Model object. Alternatively, for convenience, one of the following string values can be
            provided: "linear", "quad", "poly" or "logit", generating a LinearModel, QuadraticModel, PolynomialModel
            or "Logit" model. If  "logit" is used, then this will default to a five parameter logistic fit with
            default starting parameters (see CytoPy.flow.ext_tools.Logit for details).
        transform: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the dependent variable (standard measurements) and the independent variable (standard concentrations)
            prior to fitting the function
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        starting_params: dict, optional
            Staring parameters for chosen function. If not provided, default starting values will be used
            depending on the given model. If parameters hints have been defined this will overwrite those values.
        guess_start_params: bool (default=True)
            If True, will attempt to guess the optimal starting parameters
        model_init_kwargs: dict, optional
            Optional additional keyword arguments to pass if 'model' is of type String. Default models will be
            initialised with the given parameters.
        kwargs:
            Additional keyword arguments to pass to Model.fit call

        Returns
        -------
        None
        """
        if isinstance(model, str):
            model = default_models(model=model, model_init_kwargs=model_init_kwargs)
        if isinstance(analyte, str):
            self._fit(model, transform, analyte, starting_params, guess_start_params, **kwargs)
        else:
            for analyte in progress_bar(self.analytes):
                self._fit(model, transform, analyte, starting_params, guess_start_params, **kwargs)

    def _predict(self,
                 analyte: str):
        """
        Under the assumption of an existing standard curve, predict the concentration of the
        desired analyte (which must be present as a variable in raw data) using the standard curve.
        Results are stored as a numpy array in predictions.

        Parameters
        ----------
        analyte: str

        Returns
        -------
        None
        """
        x = self.raw[~self.raw.Sample.isin(self.standards)][[analyte]]
        transform = self.standard_curves[analyte].get("transform")
        if transform:
            x = apply_transform(x, features_to_transform=[analyte], transform_method=transform)[analyte].values
        else:
            x = x[analyte].values
        yhat = self.standard_curves[analyte].get("model_result").eval(x=x)
        if np.isnan(yhat).any():
            warn("One or more predicted values are Null; will be replaced with zeros")
        self._predictions[analyte] = np.nan_to_num(yhat)

    @assert_fitted
    def predict(self,
                analyte: str or None = None):
        """
        Under the assumption of an existing standard curve, predict the concentration of the
        desired analyte (which must be present as a variable in raw data) using the standard curve.
        Results are stored as a numpy array in predictions.

        Parameters
        ----------
        analyte

        Returns
        -------

        Raises
        ------
        AssertionError
            If standard curve has not been calculated will raise an error.

        """
        if isinstance(analyte, str):
            self._predict(analyte)
        else:
            for analyte in progress_bar(self.analytes):
                self._predict(analyte)

    def fit_predict(self,
                    model: Model or str,
                    transform: str or None = None,
                    analyte: str or None = None,
                    starting_params: dict or None = None,
                    guess_start_params: bool = True,
                    model_init_kwargs: dict or None = None,
                    **kwargs):
        """
        Calculate standard curve for the chosen analyte (see fit method for details) and predict
        (see predict method for details) concentrations. Predictions are stored to predictions attribute.

        Parameters
        ----------
        model: Model or str
            A valid lmfit.Model object. Alternatively, for convenience, one of the following string values can be
            provided: "linear", "quad", "poly" or "logit", generating a LinearModel, QuadraticModel, PolynomialModel
            or "Logit" model. If  "logit" is used, then this will default to a five parameter logistic fit with
            default starting parameters (see CytoPy.flow.ext_tools.Logit for details).
        transform: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the dependent variable (standard measurements) and the independent variable (standard concentrations)
            prior to fitting the function
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        starting_params: dict, optional
            Staring parameters for chosen function. If not provided, default starting values will be used
            depending on the given model. If parameters hints have been defined this will overwrite those values.
        guess_start_params: bool (default=True)
            If True, will attempt to guess the optimal starting parameters
        model_init_kwargs: dict, optional
            Optional additional keyword arguments to pass if 'model' is of type String. Default models will be
            initialised with the given parameters.
        kwargs:
            Additional keyword arguments to pass to scipy.optimise.curve_fit function

        Returns
        -------
        None
        """
        self.fit(model=model,
                 transform=transform,
                 analyte=analyte,
                 starting_params=starting_params,
                 guess_start_params=guess_start_params,
                 model_init_kwargs=model_init_kwargs,
                 **kwargs)
        self.predict(analyte=analyte)

    @staticmethod
    def _inverse_log(analyte: str,
                     xx: np.ndarray,
                     yhat: np.ndarray,
                     data: pd.DataFrame,
                     transform: str):
        """
        Given an analyte that has had some log transform applied apply the inverse to
        return values on a linear scale

        Parameters
        ----------
        analyte: str
        xx: Numpy.Array
        yhat
        data
        transform

        Returns
        -------

        """
        xx = list(map(INVERSE_LOG.get(transform), xx))
        yhat = list(map(INVERSE_LOG.get(transform), yhat))
        data[analyte] = data[analyte].apply(INVERSE_LOG.get(transform))
        data["conc"] = data["conc"].apply(INVERSE_LOG.get(transform))
        return xx, yhat, data

    @assert_fitted
    def plot_standard_curve(self,
                            analyte: str,
                            scatter_kwargs: dict or None = None,
                            line_kwargs: dict or None = None,
                            ax: plt.Axes or None = None):
        ax = ax or plt.subplots(figsize=(8, 8))[1]
        scatter_kwargs = scatter_kwargs or dict(facecolor="white",
                                                edgecolor="k",
                                                s=30,
                                                alpha=1,
                                                zorder=2)
        line_kwargs = line_kwargs or dict(zorder=1)
        data = self._prepare_standards_data(analyte=analyte,
                                            transform=self.standard_curves.get(analyte).get("transform"))
        xx = np.linspace(data[analyte].min() - (data[analyte].min() * 0.01),
                         data[analyte].max() + (data[analyte].max() * 0.01))
        yhat = self.standard_curves[analyte].get("model_result").eval(x=xx)
        applied_transform = self.standard_curves.get(analyte).get("transform")
        if applied_transform in ["log", "log2", "log10"]:
            xx, yhat, data = self._inverse_log(analyte=analyte,
                                               xx=xx,
                                               yhat=yhat,
                                               data=data,
                                               transform=applied_transform)
        ax.scatter(data[analyte], data["conc"], **scatter_kwargs)
        ax.plot(xx, yhat, "black", **line_kwargs)
        base = {"log": np.e,
                "log2": 2,
                "log10": 10}
        if applied_transform in ["log", "log2", "log10"]:
            ax.set_xscale("log", basex=base.get(applied_transform))
            ax.set_yscale("log", basey=base.get(applied_transform))
        ax.set_xlabel("Response")
        ax.set_ylabel("Concentration")
        return ax

    @assert_fitted
    def coef_var(self,
                 analyte: str,
                 linear_scale: bool = True):
        x = self.predictions
        if linear_scale:
            x = self.predictions_linear
        x = (x.groupby("Sample")[analyte].std()/x.groupby("Sample")[analyte].mean()).reset_index()
        return x.sort_values(analyte)

    @assert_fitted
    def plot_repeat_measures(self,
                             analyte: str,
                             log_axis: bool = True,
                             ax: plt.Axes or None = None,
                             mask: pd.DataFrame or None = None,
                             **kwargs):
        ax = ax or plt.subplots(figsize=(7, 7))[1]
        if analyte not in self.predictions.columns:
            self.predict(analyte=analyte)
        x = self.predictions
        if mask is not None:
            x = x[mask].copy()
        if log_axis:
            x = self.predictions_linear
        x["Duplication index"] = x.groupby("Sample").cumcount() + 1
        ax = pingouin.plot_paired(data=x,
                                  dv=analyte,
                                  within="Duplication index",
                                  subject="Sample",
                                  boxplot=False,
                                  ax=ax,
                                  colors=['grey', 'grey', 'grey'],
                                  **kwargs)
        if log_axis:
            ax.set_yscale("log")
        return ax

    @assert_fitted
    def plot_shift(self,
                   analyte: str,
                   factor: str,
                   **kwargs):
        assert factor in self.raw.columns, "Factor is not a valid variable. You can generate meta variables with the " \
                                           "'load_meta' function"
        assert self.raw[factor].nunique() == 2, "Factor must be binary"
        if analyte not in self.predictions.columns:
            self.predict(analyte=analyte)
        df = self.predictions
        factor_values = df[factor].unique()
        x = df[df[factor] == factor_values[0]][analyte].values
        y = df[df[factor] == factor_values[1]][analyte].values
        return pingouin.plot_shift(x, y, **kwargs)

    @assert_fitted
    def corr_matrix(self,
                    method="spearman",
                    mask: pd.DataFrame or None = None,
                    **kwargs):
        df = self.predictions
        if mask is not None:
            df = df[mask].copy()
        df = df[["Sample"] + [x for x in self._predictions.keys()]]
        corr = df.groupby("Sample").mean().corr(method=method)
        return sns.clustermap(data=corr, **kwargs)

    @assert_fitted
    def plot_box_swarm(self,
                       analyte: str,
                       factor: str):
        assert factor in self.raw.columns, "Factor is not a valid variable. You can generate meta variables with the " \
                                           "'load_meta' function"
        if analyte not in self.predictions.columns:
            self.predict(analyte=analyte)

    @assert_fitted
    def volcano_plot(self,
                     factor: str,
                     stat: str or None = None,
                     eff_size: str or None = None):
        pass

    def load_meta(self):
        pass

    @assert_fitted
    def statistics(self,
                   factor: str):
        pass
