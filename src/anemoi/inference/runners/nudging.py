# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from typing import Any

import numpy as np
import torch

from . import runner_registry
from anemoi.inference.inputs import create_input
from anemoi.utils.config import DotDict
from .default import DefaultRunner

LOG = logging.getLogger(__name__)


@runner_registry.register("nudging")
class NudgingRunner(DefaultRunner):
    """Runner for nudging (relaxing) pressure level variables
    of an anemoi model to a reference state.

    The nudging weight G relates to the relaxation time scale tau as follows:
    tau = - 1 / ln(1-G),
    where tau is in units of the anemoi model time step.
    Example: For G = 0.5 and anemoi model time step of 1 day, tau is ca. 1.5 days.

    Assumption: q, t, u, v, w, z are prognostic variables on the same pressure levels
    """

    def __init__(self, config):
        """Initialize the NudgingRunner"""
        super().__init__(config)
        try:
            self.cfg = DotDict(self.development_hacks["nudging"])
            if self.cfg.enabled:
              self._check_nudging_config()
        except KeyError:
            LOG.warning("Nudging configuration not found in development_hacks.")
            self.cfg = DotDict()
            self.cfg.enabled = False

    def _check_nudging_config(self):
        """Sanity checks on nudging configuration."""
        box = self.cfg.box

        # sanity checking on level, lat, lon configuration
        if box.lat.min < -90.0 or box.lat.max > 90.0:
            raise ValueError("Latitude values must be within -90 to 90 degrees.")
        if box.lon.min < 0.0 or box.lon.max > 360.0:
            raise ValueError("Longitude values must be within 0 to 360 degrees.")
        if box.lon.min >= box.lon.max:
            raise ValueError("Longitude min must be less than longitude max.")
        if box.lat.min >= box.lat.max:
            raise ValueError("Latitude min must be less than latitude max.")
        if any([box.lev.min, box.lev.max]) < 0:
            raise ValueError("Pressure level value must be non-negative.")
        if any([box.lev.min, box.lev.max]) > 1000:
            raise ValueError("Pressure level value must be less than 1000.")
        if box.lev.min >= box.lev.max:
            raise ValueError("Level min must be less than level max.")

        # check variable nudging coefficients
        for var in self.cfg.coeff.keys():
            coeff = self.cfg.coeff[var]
            if coeff < 0.0 or coeff > 1.0:
                raise ValueError(
                    f"Nudging coefficient for variable '{var}' must be between 0 and 1."
                )

    def get_box_weights(self, input_state):
        """Define weights for longitude-latitude-level box."""

        def add_taper(weights, x, min, max, taper):
            """Create tapering for longitudes or latitudes."""
            # tapering at eastern or northern boundary
            weights = np.where(
                np.logical_and(x > max, x < max + taper),
                (max + taper - x) / taper,
                weights,
            )
            # tapering at western or southern boundary
            weights = np.where(
                np.logical_and(x < min, x > min - taper), 1 - (min - x) / taper, weights
            )
            return weights

        box = self.cfg.box
        # Define level weights
        levellist = list()
        for var in input_state["fields"].keys():
            if var.startswith("t_"):
                level = int(var.split("_")[1])
                levellist.append(level)
        levels = np.array(sorted(levellist))

        lev_weights = np.where(
            np.logical_and(levels <= box.lev.max, levels >= box.lev.min), 1.0, 0.0
        )
        lev_weight_dict = dict(zip(levels, lev_weights))

        # Define latitude factor of weights
        lat = np.array(input_state["latitudes"])
        lat_weights = np.where(
            np.logical_and(lat >= box.lat.min, lat <= box.lat.max), 1.0, 0.0
        )
        if box.lat.taper > 0.0:
            lat_weights = add_taper(
                lat_weights, lat, box.lat.min, box.lat.max, box.lat.taper
            )

        # Define longitude factor of weights (assume 0 <= lon <= 360)
        lon = np.array(input_state["longitudes"])
        lon_weights = np.where(
            np.logical_and(lon >= box.lon.min, lon <= box.lon.max), 1.0, 0.0
        )

        if box.lon.taper > 0.0 and not (box.lon.min == 0.0 and box.lon.max == 360.0):
            if box.lon.min - box.lon.taper < 0.0 or box.lon.max + box.lon.taper > 360.0:
                raise NotImplementedError(
                    "Longitude tapering across zero meridian not implemented."
                )
            lon_weights = add_taper(
                lon_weights, lon, box.lon.min, box.lon.max, box.lon.taper
            )

        lonlat_weights = lon_weights * lat_weights

        LOG.info("Nudging active in region:")
        LOG.info(
            f"  Longitudes: {box.lon.min} to {box.lon.max} deg E with {box.lon.taper} deg taper"
        )
        LOG.info(
            f"  Latitudes: {box.lat.min} to {box.lat.max} deg N with {box.lat.taper} deg taper"
        )

        if getattr(self.cfg, "debug", False):
            import xarray as xr

            LOG.info(f"Level weights: {lev_weight_dict}")
            lonlat = list()
            for i in range(len(lon)):
                lonlat.append((lon[i], lat[i]))
            ds = xr.Dataset(
                data_vars=dict(
                    weights=("idx", lonlat_weights), lon=("idx", lon), lat=("idx", lat)
                )
            )
            ds = ds.assign_attrs(
                lon_min=box.lon.min,
                lon_max=box.lon.max,
                lon_taper=box.lon.taper,
                lat_min=box.lat.min,
                lat_max=box.lat.max,
                lat_taper=box.lat.taper,
            )
            ds.to_netcdf("~/tmp/nudging_weights.nc")

        return lev_weight_dict, lonlat_weights

    def get_var_weights(self, input_state, lev_weights):
        """Define weights for each variable based on level weights and variable coefficients."""
        var_weights = dict()
        for var, _ in input_state["fields"].items():
            for k in self.cfg.coeff.keys():
                if var.startswith(f"{k}_"):
                    name, levelstr = var.split("_")
                    level = int(levelstr)
                    if lev_weights[level] > 0.0 and self.cfg.coeff[name] > 0.0:
                        var_weights[var] = self.cfg.coeff[name] * lev_weights[level]
                        LOG.info(
                            f"Nudging pressure-level variable '{var}' with weight {var_weights[var]:.2f}"
                        )
                elif k == var and var != 'z':
                    var_weights[var] = self.cfg.coeff[var]
                    LOG.info(
                        f"Nudging surface variable '{var}' with weight {var_weights[var]:.2f}"
                    )

        return var_weights

    def forecast(self, lead_time, input_tensor_numpy, input_state):
        """Forecast the future states while nudging to a reference solution."""

        state_generator = super().forecast(lead_time, input_tensor_numpy, input_state)

        if self.cfg.enabled:
            LOG.info("Nudging to input dataset is enabled.")
            lev_weights, lonlat_weights = self.get_box_weights(input_state)
            var_weights = self.get_var_weights(input_state, lev_weights)
            config = self._input_forcings("prognostic_input", "input")
            input = create_input(self, config, variables=list(var_weights.keys()), purpose="nudging")
            for s in state_generator:
                ref_state = input.load_forcings_state(
                    dates=[s["date"]], current_state={}
                )
                for var, values in ref_state["fields"].items():
                    fref = torch.from_numpy(values).to(self.device)
                    fpred = s["fields"][var]
                    weights = torch.from_numpy(var_weights[var] * lonlat_weights).to(
                        self.device
                    )
                    fnudged = fpred - weights * (fpred - fref)
                    s["fields"][var] = torch.squeeze(fnudged, dim=(0,1))
                LOG.info(f"Nudging applied for step ending {s['date']}")
                yield s
        else:
            for s in state_generator:
                yield s
