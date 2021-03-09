#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The assay_tools model is a one-stop-shop for the analysis of plate based assays such as ELISAs, where
the first step to analysis to fitting a standard curve to infer the concentration of samples with
unknown values.

The tools present in this module assume that two DataFrames are available, referred to as 'response' and
'concentrations' throughout. These DataFrames should be formatted as follows and could be derived from an excel
document or csv file using the Pandas read_excel or read_csv method, respectively:

* response - contains the OD or MFI etc of standards, background, and any experimental samples. There should be
a column named 'Sample' that contains a unique identifier for each row. This should include identifiers for rows
that correspond to standards and background. Subsequent columns should be analytes measured.
* concentrations - contains the standard concentrations for each standard control. There should be column name d
'analyte' who's contents are the analytes measured in the response table. Subsequent columns should be
the names of measured standards, who's contents are the concentration of the standard for the given analyte (row)
and given standard (column)

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

from warnings import warn
from cytopy.flow.transform import apply_transform
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import uncertainties as unc
import seaborn as sns
import pandas as pd
import numpy as np

np.seterr(over="raise")


def subtract_background(data: pd.DataFrame,
                        background_id: str,
                        analytes: list):
    """
    Given a DataFrame of assay data subtract background using the rows with a Sample ID corresponding to the
    'background_id'. Assumes the DataFrame is wide (columns correspond to analytes/variables) and has an
    identifier column named 'Sample'.

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


def wrangle_standard_data(analyte: str,
                          response: pd.DataFrame,
                          concentrations: pd.DataFrame,
                          transformations: dict or None = None,
                          standard_regex: str = "Standard[0-9]+"):
    """
    Given the response dataframe and standard concentrations dataframe (see documentation for examples),
    wrangle into a single dataframe that summarises the assay standard controls. The resulting dataframe
    will have three columns: Sample, Concentration, and Response.

    Parameters
    ----------
    analyte: str
        Name of the analyte of interest. The response dataframe will be filtered to collect the response
        values for this analyte (e.g. OD or MFI) for standard controls
    response: Pandas.DataFrame
        Response dataframe (see cytopy.assay_tools documentation for examples)
    concentrations: Pandas.DataFrame
        Concentrations dataframe (see cytopy.assay_tools documentation for examples)
    transformations: dict, optional
        Provide a dictionary with the keys "Concentration" and "Response". The value of the each should
        correspond to a desired transformation to be applied (see cytopy.flow.transforms for available transforms)
    standard_regex: str (default="Standard[0-9]+")
        Regular expression used to identify rows corresponding to standard controls. Searches the "Sample" column
        and by default looks for the term 'Standard' suffixed with a single number. Is case sensitive.

    Returns
    -------
    Pandas.DataFrame
    """
    assert "Sample" in response.columns, "Sample column missing from response DataFrame"
    standards = list(response.Sample[response.Sample.str.contains(standard_regex, case=True, regex=True)].unique())
    assert len(standards) > 0, "No sample IDs match the given standard_regex expression"
    assert all([x in concentrations.columns for x in standards]), f"One or more standards {standards} missing " \
                                                                  f"from concentrations DataFrame"
    x = (response[response.Sample.isin(standards)][["Sample", analyte]]
         .copy()
         .rename({analyte: "Response"}, axis=1))
    y = (concentrations[concentrations.analyte == analyte]
         .copy()
         .melt(var_name="Sample", value_name="Concentration"))
    standards = x.merge(y, on="Sample")
    standards["Response"] = standards["Response"].astype(dtype="float64")
    standards["Concentration"] = standards["Concentration"].astype(dtype="float64")
    if transformations is not None:
        return apply_transform(data=standards, features_to_transform=transformations)
    return standards


def generalised_hill_equation(concentration: np.ndarray,
                              a: float,
                              d: float,
                              slope: float,
                              log_inflection_point: float,
                              symmetry: float = 1.0):
    """
    Five parameter logistic fit for dose-response curves. If four parameter fit is desired, symmetry should have a
    value of 1.0.

    Paul G. Gottschalk, John R. Dunn, The five-parameter logistic: A characterization and comparison with the
    four-parameter logistic, Analytical Biochemistry, Volume 343, Issue 1, 2005 https://doi.org/10.1016/j.ab.2005.04.035

    Parameters
    ----------
    concentration: numpy.ndarray
        X-axis variable
    a: float
        Bottom asymptote
    d: float
        Top asymptote
    slope: float
        Steepness of the curve at the inflection point
    log_inflection_point: float
        Inflection point as read on the x-axis (scale equivalent to dose)
    symmetry: float
        Degree of asymmetry

    Returns
    -------
    numpy.ndarray
    """
    assert slope > 0, "parameter 'slope' must be greater than 0"
    assert symmetry > 0, "parameter 'symmetry' must be greater than 0"
    concentration = np.log10(concentration)
    numerator = a - d
    denominator = (1 + (concentration / log_inflection_point) ** slope) ** symmetry
    return d + (numerator / denominator)


def inverse_generalised_hill_equation(response: np.ndarray,
                                      a: float,
                                      d: float,
                                      slope: float,
                                      log_inflection_point: float,
                                      symmetry: float = 1.0):
    """
    Inverse of the five parameter logistic fit for dose-response curves.
    If four parameter fit is desired, symmetry should have a value of 1.0.

    Paul G. Gottschalk, John R. Dunn, The five-parameter logistic: A characterization and comparison with the
    four-parameter logistic, Analytical Biochemistry, Volume 343, Issue 1, 2005 https://doi.org/10.1016/j.ab.2005.04.035

    Parameters
    ----------
    response: numpy.ndarray
        Y-axis variable
    a: float
        Bottom asymptote
    d: float
        Top asymptote
    slope: float
        Steepness of the curve at the inflection point
    log_inflection_point: float
        Inflection point as read on the x-axis (scale equivalent to dose)
    symmetry: float
        Degree of asymmetry

    Returns
    -------
    numpy.ndarray
    """
    assert slope > 0, "parameter 'slope' must be greater than 0"
    assert symmetry > 0, "parameter 'symmetry' must be greater than 0"
    xi = ((a - d) / (response - d))
    xi = np.where((xi < 1.0) & (xi > 0.0), 1.0, xi)
    return 10 ** (log_inflection_point * ((xi ** (1 / symmetry)) - 1) ** (1 / slope))


def estimate_inflection_point(concentration: np.ndarray,
                              response: np.ndarray):
    """
    Estimate the optimal inflection point based on the stepwise distance between response
    values on a linear scale. Where the greatest increase in response is observed, the index
    is used to select the concentration, which is returned as a float.

    Parameters
    ----------
    concentration: numpy.ndarray
    response: numpy.ndarray

    Returns
    -------
    Numpy.float64
    """
    dist = list()
    for i in range(response.shape[0] - 1):
        dist.append(np.abs(response[i] - response[i + 1]))
    i = np.argmax(dist) + 1
    return np.log10(concentration[i])


def rsquared(func: callable,
             params: np.ndarray,
             conc: np.ndarray,
             response: np.ndarray):
    """
    Calculate R-squared for a given function when fitted to concentrations (conc) with intent
    to predict response.

    Parameters
    ----------
    func: callable
    params: numpy.ndarray
        Estimated optimal parameters
    conc: numpy.ndarray
    response: numpy.ndarray

    Returns
    -------
    Numpy.float64
    """
    err = (response - func(conc, *params))
    sse = np.sum(err ** 2)
    var = (len(response) - 1.0) * np.var(response, ddof=1)
    return 1.0 - (sse / var)


def confidence_band(xspace: np.ndarray,
                    conc: np.ndarray,
                    response: np.ndarray,
                    optimal_parameters: np.ndarray,
                    func: callable,
                    alpha: float = 0.05):
    """
    Generate bivariate confidence bands for some regressor (func).

    Parameters
    ----------
    xspace: numpy.ndarray
        Linear space in which the regression curve is to be plotted
    conc: numpy.ndarray
        Concentrations (x-axis) variable
    response: numpy.ndarray
        Response (y-axis) variable
    optimal_parameters: numpy.ndarray
        Estimated optimal parmameters for func
    func: callable
        Some regression function
    alpha: float (default=0.05)
        Significance level

    Returns
    -------

    """
    sample_size = len(conc)
    n_params = len(optimal_parameters)
    studentt = stats.t.ppf(1.0 - alpha, sample_size - n_params)
    sse = np.sum((response - func(conc, *optimal_parameters)) ** 2)
    std = np.sqrt(1.0 / (sample_size - n_params) * sse)

    sx = (xspace - conc.mean()) ** 2
    sxd = np.sum((conc - conc.mean()) ** 2)

    yhat = func(xspace, *optimal_parameters)
    dy = studentt * std * np.sqrt(1.0 + (1.0 / sample_size) + (sx / sxd))
    return yhat - dy, yhat + dy


class LogisticCurveFit:
    """
    Logistic curve fitting model with either four or five parameters using the generalised hill equation.
    By default, uses five parameter fit unless user specifies 'four_parameter_fit = True' when constructing
    the object (in which case, the 'symmetry' parameter is bound to a value of 1).

    Starting parameters are stored within the 'starting_params' attribute which is an array of values
    corresponding to the parameters of the generalised hill equation in this order: 'a', 'd', 'slope',
    'log_inflection_point' and 'symmetry'. See cytopy.assay_tools.generalised_hill_equation for details.

    Set the starting parameters by writing to this attribute or calling the 'set_starting_params' method, which
    will estimate optimal starting parameters using input data. This will also estimate parameter bounds. By default,
    the starting parameter bounds are:
    - 'a' (top asymptote): -inf, inf
    - 'd' (top asymptote): -inf, inf
    - 'slope': 0, inf
    - 'log_inflection_point': -inf, inf
    - 'symmetry': 0.001, 10

    Both 'slope' and 'symmetry' must always have a lower bound of 0, any value below this will raise an AssertionError
    when calling 'fit'. Parameter bounds are stored in the 'parameter_bounds' attribute as a tuple of two Numpy Arrays.
    The first of these arrays are the lower bounds and the second the upper bounds. The order in each array must match
    the order in 'starting_params'. When calling 'set_starting_params', the upper bound of 'slope' and the lower bound
    of 'a' are defined, but can be overwritten by either writing to 'parameter_bounds' directly, or by adjusting
    the parameters of 'set_starting_params'.

    Attributes
    -----------
    starting_params: numpy.ndarray
        Starting parameters passed to generalised hill equation during curve fitting
    optimal_params: numpy.ndarray
        Optimal parameters estimated using least squares regression and populated by the 'fit' method
    pcov: numpy.ndarray
        The estimated covariance of optimal parameters
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    perr: numpy.ndarray
        Standard errors on the estimated parameters
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    parameter_bounds: tuple (numpy.ndarray, numpy.ndarray)
        The first of these arrays are the lower bounds and the second the upper bounds. The order in each array must match
        the order in 'starting_params'
    rsquared: float
        R^2 value for fit (populated by 'fit' method). See cytopy.assay_tools.rsquared.
    parameter_unc: numpy.ndarray
        Parameter confidence intervals
    """

    def __init__(self,
                 starting_params: list or None = None,
                 parameter_bounds: list or None = None,
                 four_parameter_fit: bool = False):
        self._starting_params = starting_params or np.array([])
        self.optimal_params = None
        self.pcov = None
        self.perr = None
        self._training_data = None
        self.parameter_bounds = parameter_bounds or (np.array([-np.inf, -np.inf, 0, -np.inf, 0.001]),
                                                     np.array([np.inf, np.inf, np.inf, np.inf, 10]))
        self.rsquared = None
        self.fourpl = four_parameter_fit
        self.parameter_unc = None
        if four_parameter_fit:
            self.parameter_bounds[0][4] = 1.0
            self.parameter_bounds[1][4] = 1.0

    @property
    def starting_params(self):
        return self._starting_params

    @starting_params.setter
    def starting_params(self, values: np.ndarray):
        assert len(values) == 5, "values should be a numpy array with exactly 5 elements"
        self._starting_params = values

    def set_staring_params(self,
                           conc: np.ndarray,
                           response: np.ndarray,
                           a_min: float or None = None,
                           slope_max: float or None = None,
                           **kwargs):
        """
        In a data-driven fashion, defines the starting parameters and the bounds for 'a' and 'slope'. Starting
        parameters are defined like so:
        * a = 1e=-5
        * d = max(response)
        * log_inflection_point = estimate_inflection_point(conc, response)
          See cytopy.assay_tools.estimate_inflection_point for details
        * slope = 10th percentile of linear space between min(response) and max(response)
        * symmetry = 0.5

        Any of the above can be overwritten by passing the parameter name and value to kwargs.

        In addition, the lower bound of 'a' is set to equal min(response) but can be overwritten by providing a value
        to 'a_min'. In the same vain, the maximum bound for 'slope' is set to equal max(response) * 0.1, but can
        be overwritten by providing a value to 'slope_max'

        Parameters
        ----------
        conc: numpy.ndarray
            Concentration (x-axis variable)
        response: numpy.ndarray
            Response (y-axis variable)
        a_min: float, optional
            Overwrites minimum bound for 'a'
        slope_max: float, optional
            Overwrites maximum bound for 'slope'
        kwargs:
            Overwrite any parameter starting value using kwargs

        Returns
        -------
        None
        """
        a = kwargs.get("a", 1e-5)
        d = kwargs.get("d", np.max(response))
        log_inflection_point = kwargs.get("log_inflection_point", estimate_inflection_point(conc, response))
        slope = kwargs.get("slope", np.percentile(np.linspace(np.min(response), np.max(response), 1000), 0.1))
        symmetry = kwargs.get("symmetry", 0.5)
        if self.fourpl:
            symmetry = 1.0
        self._starting_params = [a, d, slope, log_inflection_point, symmetry]
        # Bound 'a' (lower asymptote) so it's smallest possible value is min(response) or user defined
        self.parameter_bounds[1][0] = a_min or np.min(response)
        # Bound 'slope' so it's largest possible value is max(response) * 0.1 or user defined
        self.parameter_bounds[1][2] = slope_max or np.max(response) * 0.1

    def fit(self,
            conc: np.ndarray,
            response: np.ndarray,
            maxfev=15000,
            **kwargs):
        """
        Fit data using the generalised hill equation. Optimal parameters are estimated using least squares regression
        (Scipy.optimise.curve_fit) and stored in the attribute 'optimal_params'.

        Parameters
        ----------
        conc: numpy.ndarray
            Concentration variable (x-axis)
        response: numpy.ndarray
            Response variable (x-axis)
        maxfev: int (default=15000)
            Maximum number of epochs (fits) to attempt
        kwargs:
            Additional keyword arguments to pass to scipy.optimise.curve_fit

        Returns
        -------
        None
        """
        if len(self._starting_params) == 0:
            self.set_staring_params(conc=conc, response=response)
        self.optimal_params, self.pcov = curve_fit(generalised_hill_equation, conc, response,
                                                   p0=self._starting_params,
                                                   maxfev=maxfev, **kwargs,
                                                   bounds=self.parameter_bounds)
        self.perr = np.sqrt(np.diag(self.pcov))
        self._training_data = dict(x=conc, y=response)
        self.rsquared = rsquared(func=generalised_hill_equation,
                                 params=self.optimal_params,
                                 conc=conc,
                                 response=response)
        self.parameter_unc = unc.correlated_values(self.optimal_params, self.pcov)

    def predict(self, response: np.ndarray):
        """
        Using the optimal parameters estimated during 'fit', use the inverse generalised hill equation
        to estimate the concentrations for some given response.

        Parameters
        ----------
        response: numpy.ndarray

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        AssertionError
            If 'fit' has not previously been called
        """
        assert self.optimal_params is not None, "Must call 'fit' prior to 'predict'"
        predictions = inverse_generalised_hill_equation(response, *self.optimal_params)
        if np.isnan(predictions).any():
            warn("One or more values exceeds detectable range. Null values will be replaced with "
                 "a value twice the highest standard control")
        idx = np.where((predictions == 1.0) | np.isnan(predictions))[0]
        predictions = np.nan_to_num(predictions, nan=np.max(self._training_data.get("x")) * 2)
        return predictions, idx

    def standard_curve(self,
                       ax: plt.Axes or None = None,
                       fit_response: np.ndarray or None = None,
                       log_y: bool = False,
                       curve_kwargs: dict or None = None,
                       scatter_kwargs: dict or None = None,
                       fit_scatter_kwargs: dict or None = None,
                       overwrite_params: list or None = None):
        """
        Plot the standard curve of the fitted generalised hill function.

        Parameters
        ----------
        ax: Matplotlib.axes.Axes, optional
            Axes to plot data on. If not provided, a figure will be generated of size 5x5
        fit_response: numpy.ndarray, optional
            If a numpy array is provided, it is assumed to be the response of some sample with unknown
            concentrations. The concentrations will be predicted using the 'predict' method and plotted
            on the standard curve.
        log_y: bool (default=True)
            Plot y-axis on a log10 scale
        curve_kwargs: dict, optional
            Keyword arguments for the line plot of the standard curve (Axes.plot)
        scatter_kwargs: dict, optional
            Keyword arguments for the scatter plot of the standard controls (Axes.scatter)
        fit_scatter_kwargs: dict, optional
            Keyword arguments for the scatter plot of the overlayed predicted concentrations (Axes.scatter). Ignored
            if 'fit_response' is None.
        overwrite_params: dict, optional
            If provided, will overwrite the optimal parameters when fitting the curve.

        Returns
        -------
        Matplotlib.axes.Axes
        """
        curve_kwargs = curve_kwargs or {}
        scatter_kwargs = scatter_kwargs or {}
        fit_scatter_kwargs = fit_scatter_kwargs or {}
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        assert self.optimal_params is not None, "Call 'fit' prior to calling 'standard_curve'"
        assert self._training_data is not None, "Call 'fit' prior to calling 'standard_curve'"
        start = np.min(self._training_data.get("x"))
        end = np.max(self._training_data.get("x"))
        xx = np.linspace(start, end, 1000)
        params = overwrite_params or self.optimal_params
        yhat = generalised_hill_equation(xx, *params)
        ax.plot(xx, yhat,
                zorder=curve_kwargs.pop("zorder", 1),
                color=curve_kwargs.pop("color", "black"))
        ax.scatter(self._training_data.get("x"),
                   self._training_data.get("y"),
                   facecolor=scatter_kwargs.pop("facecolor", "white"),
                   edgecolor=scatter_kwargs.pop("edgecolor", "k"),
                   s=scatter_kwargs.pop("s", 50),
                   alpha=scatter_kwargs.pop("alpha", 1),
                   zorder=curve_kwargs.pop("zorder", 2))
        if fit_response is not None:
            xhat, err_idx = self.predict(response=fit_response)
            ax.scatter(np.delete(xhat, err_idx),
                       np.delete(fit_response, err_idx),
                       facecolor=fit_scatter_kwargs.pop("facecolor", "black"),
                       marker=fit_scatter_kwargs.pop("marker", "x"),
                       s=fit_scatter_kwargs.pop("s", 35),
                       zorder=fit_scatter_kwargs.pop("zorder", 3))
            ax.scatter(xhat[err_idx],
                       fit_response[err_idx],
                       facecolor=fit_scatter_kwargs.pop("facecolor", "red"),
                       marker=fit_scatter_kwargs.pop("marker", "x"),
                       s=fit_scatter_kwargs.pop("s", 35),
                       zorder=fit_scatter_kwargs.pop("zorder", 3))
        ax.set_xscale("log", base=10)
        if log_y:
            ax.set_yscale("log", base=10)
        return ax


def linear_eq(concentration: np.ndarray,
              slope: float,
              bias: float):
    return concentration * slope + bias


def inverse_linear_eq(response: np.ndarray,
                      slope: float,
                      bias: float):
    return (response - bias) / slope


class LinearFit:
    """
    Fit straight line function to dose-response data using least squares regression (scipy.optimize.curve_fit)

    Attributes
    -----------
    starting_params: numpy.ndarray
    optimal_params: numpy.ndarray
        Optimal parameters estimated using least squares regression and populated by the 'fit' method
    pcov: numpy.ndarray
        The estimated covariance of optimal parameters
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    perr: numpy.ndarray
        Standard errors on the estimated parameters
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    parameter_bounds: tuple (numpy.ndarray, numpy.ndarray)
        The first of these arrays are the lower bounds and the second the upper bounds. The order in each array must match
        the order in 'starting_params'
    """

    def __init__(self,
                 starting_params: list or None = None,
                 parameter_bounds: list or None = None):
        self._starting_params = starting_params
        self.optimal_params = np.array([])
        self.pcov = None
        self.perr = None
        self._training_data = None
        self._log_conc = False
        self._log_response = False
        self.parameter_bounds = parameter_bounds or (np.array([-np.inf, -np.inf]),
                                                     np.array([np.inf, np.inf]))
        self.rsquared = None
        self.parameter_unc = None

    @property
    def starting_params(self):
        return self._starting_params

    @starting_params.setter
    def starting_params(self, values: np.ndarray):
        assert len(values) == 2, "values should be a numpy array with exactly 2 elements"
        self._starting_params = values

    def fit(self,
            conc: np.ndarray,
            response: np.ndarray,
            log_conc: bool = True,
            log_response: bool = True,
            maxfev=15000,
            **kwargs):
        """
        Fit data using linear equation (y=mx+c). Optimal parameters are estimated using least squares regression
        (Scipy.optimise.curve_fit) and stored in the attribute 'optimal_params'.

        Parameters
        ----------
        conc: numpy.ndarray
            Concentration variable (x-axis)
        response: numpy.ndarray
            Response variable (x-axis)
        log_conc: bool (default=True)
            Log10 transform concentrations prior to fit
        log_response: bool (default=True)
            Log10 transform response prior to fit
        maxfev: int (default=15000)
            Maximum number of epochs (fits) to attempt
        kwargs:
            Additional keyword arguments to pass to scipy.optimise.curve_fit

        Returns
        -------
        None
        """
        self._training_data = dict(x=conc, y=response)
        if log_conc:
            conc = np.log10(conc)
            self._log_conc = True
        if log_response:
            response = np.log10(response)
            self._log_response = True
        self.optimal_params, self.pcov = curve_fit(linear_eq,
                                                   conc,
                                                   response,
                                                   p0=self._starting_params,
                                                   maxfev=maxfev,
                                                   bounds=self.parameter_bounds,
                                                   **kwargs)
        self.perr = np.sqrt(np.diag(self.pcov))
        self.rsquared = rsquared(func=linear_eq,
                                 params=self.optimal_params,
                                 conc=conc,
                                 response=response)
        self.parameter_unc = unc.correlated_values(self.optimal_params, self.pcov)

    def predict(self, response: np.ndarray):
        """
        Using the optimal parameters estimated during 'fit', use the inverse linear equation (x = (y-c)/m)
        to estimate the concentrations for some given response.

        Parameters
        ----------
        response: numpy.ndarray

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        AssertionError
            If 'fit' has not previously been called
        """
        assert self.optimal_params is not None, "Must call 'fit' prior to 'predict'"
        if self._log_response:
            response = np.log10(response)
        x = inverse_linear_eq(response=response, *self.optimal_params)
        if self._log_conc:
            return 10 ** x
        return x

    def standard_curve(self,
                       ax: plt.Axes or None = None,
                       fit_response: np.ndarray or None = None,
                       log_y: bool = True,
                       curve_kwargs: dict or None = None,
                       scatter_kwargs: dict or None = None,
                       fit_scatter_kwargs: dict or None = None,
                       overwrite_params: list or None = None):
        """
        Plot the standard curve of the linear fit

        Parameters
        ----------
        ax: Matplotlib.axes.Axes, optional
            Axes to plot data on. If not provided, a figure will be generated of size 5x5
        fit_response: numpy.ndarray, optional
            If a numpy array is provided, it is assumed to be the response of some sample with unknown
            concentrations. The concentrations will be predicted using the 'predict' method and plotted
            on the standard curve.
        log_y: bool (default=True)
            Plot y-axis on a log10 scale
        curve_kwargs: dict, optional
            Keyword arguments for the line plot of the standard curve (Axes.plot)
        scatter_kwargs: dict, optional
            Keyword arguments for the scatter plot of the standard controls (Axes.scatter)
        fit_scatter_kwargs: dict, optional
            Keyword arguments for the scatter plot of the overlayed predicted concentrations (Axes.scatter). Ignored
            if 'fit_response' is None.
        overwrite_params: dict, optional
            If provided, will overwrite the optimal parameters when fitting the curve.

        Returns
        -------
        Matplotlib.axes.Axes
        """
        curve_kwargs = curve_kwargs or {}
        scatter_kwargs = scatter_kwargs or {}
        fit_scatter_kwargs = fit_scatter_kwargs or {}
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        assert self.optimal_params is not None, "Call 'fit' prior to calling 'standard_curve'"
        assert self._training_data is not None, "Call 'fit' prior to calling 'standard_curve'"

        # Generate linear space
        start = np.min(self._training_data.get("x")) - np.median(self._training_data.get("x")) * 0.1
        end = np.max(self._training_data.get("x")) + np.median(self._training_data.get("x")) * 0.1
        xx = np.linspace(start, end, 1000)

        # Estimate response using optimal parameters
        params = overwrite_params or self.optimal_params
        if self._log_conc:
            yhat = linear_eq(np.log10(xx), *params)
        else:
            yhat = linear_eq(xx, *params)
        if self._log_response:
            yhat = 10 ** yhat

        # Plot curve and standard controls
        ax.plot(xx, yhat,
                zorder=curve_kwargs.pop("zorder", 1),
                color=curve_kwargs.pop("color", "black"))
        ax.scatter(self._training_data.get("x"),
                   self._training_data.get("y"),
                   facecolor=scatter_kwargs.pop("facecolor", "white"),
                   edgecolor=scatter_kwargs.pop("edgecolor", "k"),
                   s=scatter_kwargs.pop("s", 50),
                   alpha=scatter_kwargs.pop("alpha", 1),
                   zorder=curve_kwargs.pop("zorder", 2))

        if fit_response is not None:
            if self._log_response:
                xhat = linear_eq(np.log10(fit_response), *self.optimal_params)
            else:
                xhat = linear_eq(fit_response, *self.optimal_params)
            if self._log_conc:
                xhat = 10 ** xhat
            ax.scatter(xhat,
                       fit_response,
                       facecolor=fit_scatter_kwargs.pop("color", "red"),
                       edgecolor=fit_scatter_kwargs.pop("marker", "x"),
                       s=fit_scatter_kwargs.pop("s", 35),
                       zorder=fit_scatter_kwargs.pop("zorder", 3))

        ax.set_xscale("log", base=10)
        if log_y:
            ax.set_yscale("log", base=10)
        return ax


def standards_sample_response_density(response: pd.DataFrame,
                                      concentrations: pd.DataFrame,
                                      analyte: str,
                                      xlabel: str or None = None,
                                      transformations: dict or None = None,
                                      standard_regex: str = "Standard[0-9]+",
                                      standard_colour: str = "red",
                                      standard_linestyle: str = "dotted",
                                      sample_colour: str = "black",
                                      sample_linestyle: str = "solid",
                                      ax: plt.Axes or None = None,
                                      cumulative: bool = False,
                                      plot_response: bool = True,
                                      **legend_kwargs):
    """
    Generate a KDE plot comparing the distribution of response values (e.g. OD or MFI) between the standard
    controls and your measured samples. Expects standard response and concentrations dataframe (see cytopy.assay_tools
    documentation).

    Parameters
    ----------
    response: Pandas.DataFrame
        Response dataframe i.e. OD or MFI measures for standards and samples
    concentrations: Pandas.DataFrame
        Concentrations dataframe, containing standard control concentrations. Columns should include 'analyte'
        and standard control names as found in the response Sample column (see cytopy.assay_tools.wrange_standard_data
        for details)
    analyte: str
        Analyte of interest
    xlabel: str (optional)
        X-axis label
    transformations: dict, optional
        Transformations to be applied, if any. If required, see cytopy.assay_tools.wrange_standard_data for details.
    standard_regex: str (default="Standard[0-9]+")
        See cytopy.assay_tools.wrange_standard_data
    standard_colour: str (default="red")
        Colour for KDE line for standard controls
    standard_linestyle: str, (default="dotted")
        Line style for KDE line for standard controls
    sample_colour: str, (default="black")
        Colour for KDE line for samples
    sample_linestyle: str, (default="solid")
        Line style for KDE line for samples
    ax: Matplotlib.Axes, optional
        Axis object
    cumulative: bool (default=False)
        If True, plots cumulative probability function, as opposed to probability density function
    plot_response: bool (default=True)
        If True, plots the sample KDE alongside the standards. Set to False to only observe the standard
        control KDE
    legend_kwargs:
        Additional keyword arguments to control legend settings

    Returns
    -------
    Matplotlib.Axes
    """
    xlabel = xlabel or analyte
    standards = wrangle_standard_data(response=response,
                                      concentrations=concentrations,
                                      analyte=analyte,
                                      transformations=transformations,
                                      standard_regex=standard_regex)
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    sns.kdeplot(response[~response.Sample.str.contains(standard_regex, regex=True)][analyte],
                cumulative=cumulative,
                color=sample_colour,
                linestyle=sample_linestyle,
                label="Sample distribution",
                ax=ax)
    if plot_response:
        sns.kdeplot(standards["Response"],
                    cumulative=cumulative,
                    color=standard_colour,
                    linestyle=standard_linestyle,
                    label="Standard distribution",
                    ax=ax)
    ax.legend(**legend_kwargs)
    ax.set_xlim(0)
    ax.set_xlabel(xlabel)
    return ax


def predictions_dataframe(model: LogisticCurveFit or LinearFit,
                          response: pd.DataFrame,
                          analyte: str,
                          dilution_factor: int = 1,
                          standard_regex: str = "Standard[0-9]+"):
    """
    Given a valid model object (LogisticCurveFit or LinearFit) generate a dataframe of
    predicted concentration values for samples in the response dataframe
    Parameters
    ----------
    model: LogisticCurveFit or LinearFit
    response: Pandas.DataFrame
    analyte: Pandas.DataFrame
    standard_regex: str, (default="Standard[0-9]+")
        Regular expression term used to identify rows corresponding to standard controls

    Returns
    -------
    Pandas.DataFrame
    """
    x = response[~response.Sample.str.contains(standard_regex)][analyte].values
    p, err = model.predict(x)
    flag = np.zeros(len(p))
    flag[err] = 1
    sample_ids = response[~response.Sample.str.contains("Standard")].Sample.values
    df = pd.DataFrame({"Sample": sample_ids,
                       "concentration": p,
                       "flag": flag})
    df["duplicate"] = df.groupby("Sample").cumcount() + 1
    df["analyte"] = analyte
    df["concentration"] = df["concentration"]/dilution_factor
    return df


def plot_repeat_measures(predictions_df: pd.DataFrame,
                         ax: plt.Axes or None = None,
                         log_axis: bool = True):
    """
    Given a predictions dataframe, as generated by cytopy.assay_tools.predictions_dataframe, plot
    duplicated assay results as a point plot.

    Parameters
    ----------
    predictions_df: Pandas.DataFrame
    ax: Matplotlib.Axes, optional
    log_axis: bool (default=True)
        Log10 y-axis

    Returns
    -------
    Matplotlib.Axes
    """
    ax = ax or plt.subplots(figsize=(5, 8))[1]
    max_dup = []
    for _, sample_df in predictions_df.groupby("Sample"):
        c = sample_df.concentration.values
        duplicates = list(range(len(c)))
        ax.plot(duplicates, c, color="b", zorder=1)
        ax.scatter(duplicates, c, facecolor="black", zorder=2)
        if len(duplicates) > len(max_dup):
            max_dup = duplicates
    ax.set_xlabel("Duplicates")
    ax.set_ylabel("Concentration")
    ax.set_xticks(max_dup)
    if log_axis:
        ax.set_yscale("log", base=10)
        ax.set_ylabel("log10(Concentration)")
    return ax


def rank_cv(predictions_df: pd.DataFrame):
    return predictions_df.groupby("Sample")["concentration"].apply(stats.variation).sort_values(ascending=False)
