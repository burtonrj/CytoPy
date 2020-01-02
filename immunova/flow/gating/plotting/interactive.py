from immunova.flow.gating.actions import Gating
from immunova.flow.gating.plotting.static_plots import transform_axes, PlottingError
import plotly.express as px
import pandas as pd


class InteractivePlotting:
    """
    Interactive plotting of Gating object using Plotly

    Attributes:
        gating: Gating object (see flow.gating.actions.Gating)
    """
    def __init__(self, gating: Gating):
        self.gating = gating

    def _get_data(self, population_name, fmo_x: str or None = None, fmo_y: str or None = None,
                  fmo_z: str or None = None) -> pd.DataFrame:
        """
        Fetch a Pandas DataFrame of single cell data for a given population
        :param population_name: required population
        :param fmo_x: name of FMO for x-axis dimension (optional)
        :param fmo_y: name of FMO for y-axis dimension (optional)
        :param fmo_z: name of FMO for z-axis dimension (optional)
        :return: Pandas DataFrame of single cell data for given population
        """
        assert population_name in self.gating.populations.keys(), f'Invalid population name, must be one of {self.gating.populations.keys()}'
        data = dict(primary=self.gating.get_population_df(population_name, transform=False).copy())
        for fmo in [fmo_x, fmo_y, fmo_z]:
            if fmo:
                data[fmo] = self.gating.get_fmo_data(target_population=population_name,
                                                     fmo=fmo)
        return data

    @staticmethod
    def __transform_data(data: dict, axes_vars: dict, transforms: dict):
        if transforms is None:
            transforms = dict(x='logicle', y='logicle', z='logicle')
        return {k: transform_axes(v, axes_vars, transforms) for k, v in data.items()}

    def scatter_3d(self, population_name, x, y, z, transforms):
        if population_name in self.gating.populations.keys():
            data = self.gating.get_population_df(population_name).copy()
        else:
            print(f'Invalid population name, must be one of {self.gating.populations.keys()}')
            return None
        if transforms is None:
            transforms = dict(x='logicle', y='logicle', z='logicle')
        data = transform_axes(data=data, axes_vars={'x': x, 'y': y, 'z': z}, transforms=transforms)
        fig = px.scatter_3d(data, x=x, y=y, z=z, opacity=0.4)
        fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

    def scatter_3d_fmo(self, population_name, x, y, z, transforms,
                       fmo_x: str or None = None, fmo_y: str or None = None,
                       fmo_z: str or None = None):
        data = self._get_data(population_name=population_name, fmo_x=fmo_x, fmo_y=fmo_y, fmo_z=fmo_z)
        data = self.__transform_data(data=data, axes_vars={'x': x, 'y': y, 'z': z}, transforms=transforms)
        d = pd.DataFrame()
        for k, v in data.items():
            v['origin'] = k
            d = pd.concat([d, v])
        fig = px.scatter_3d(d, x=x, y=y, z=z, opacity=0.4, color='origin')
        fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

