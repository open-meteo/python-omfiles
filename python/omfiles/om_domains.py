from typing import List

import numpy as np

from omfiles.grids import (
    AbstractGrid,
    ProjectionGrid,
    RegularLatLonGrid,
    RotatedLatLonProjection,
    StereographicProjection,
)
from omfiles.utils import EPOCH


class OmDomain:
    """
    Class representing a domain configuration for a weather model.

    This class provides metadata and configuration for different
    weather model grids used in Open-Meteo.
    """

    def __init__(
        self,
        name: str,
        grid: AbstractGrid,
        file_length: int,
        temporal_resolution_seconds: int = 3600
    ):
        """
        Initialize a domain configuration.

        Parameters:
        -----------
        name : str
            Name of the domain
        grid : AbstractGrid
            Grid implementation for this domain
        file_length : int
            Number of time steps in each file chunk
        temporal_resolution_seconds : int, optional
            Time resolution in seconds (default: 3600 = 1 hour)
        """
        self.name = name
        self.grid = grid
        self.file_length = file_length
        self.temporal_resolution_seconds = temporal_resolution_seconds

    def time_to_chunk_index(self, timestamp: np.datetime64) -> int:
        """
        Convert a timestamp to a chunk index. This depends on the file_length
        and the temporal_resolution_seconds of the domain.

        Parameters:
        -----------
        timestamp : np.datetime64
            The timestamp to convert

        Returns:
        --------
        int
            The chunk index containing the timestamp
        """
        seconds_since_epoch = (timestamp - EPOCH) / np.timedelta64(1, 's')
        chunk_index = int(seconds_since_epoch / (self.file_length * self.temporal_resolution_seconds))
        return chunk_index

    def chunks_for_date_range(
        self,
        start_timestamp: np.datetime64,
        end_timestamp: np.datetime64,
    ) -> List[int]:
        """
        Find all chunk indices that contain data within the given date range.

        Parameters:
        -----------
        start_date : datetime
            Start date for the data range
        end_date : datetime
            End date for the data range
        Returns:
        --------
        List[int]
            List of chunk indices containing data within the date range
        """
        # Get chunk indices for start and end dates
        start_chunk = self.time_to_chunk_index(start_timestamp)
        end_chunk = self.time_to_chunk_index(end_timestamp)

        # Generate list of all chunks between start and end (inclusive)
        return list(range(start_chunk, end_chunk +1))

    def get_chunk_time_range(self, chunk_index: int):
        """
        Get the time range covered by a specific chunk.

        Parameters:
        -----------
        chunk_index : int
            Index of the chunk

        Returns:
        --------
        np.ndarray
            Array of datetime64 objects representing the time points in the chunk
        """
        chunk_start_seconds = chunk_index * self.file_length * self.temporal_resolution_seconds
        start_time = EPOCH + np.timedelta64(chunk_start_seconds, 's')

        # Generate timestamps at regular intervals from the start time
        time_delta = np.timedelta64(self.temporal_resolution_seconds, 's')
        # Note: better type inference via list comprehension here
        timestamps = np.array([start_time + i * time_delta for i in range(self.file_length)])
        return timestamps

# - MARK: Create grid instances for supported domains

# DWD ICON global is regularized during download to nx: 2879, ny: 1441 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Icon/Icon.swift#L146
_dwd_icon_grid = RegularLatLonGrid(
    lat_start=-90,
    lat_steps=1441,
    lat_step_size=0.125,
    lon_start=-180,
    lon_steps=2879,
    lon_step_size=0.125
)

# DWD ICON EU is regularized during download to nx: 1377, ny: 657 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Icon/Icon.swift#L148
_dwd_icon_eu_grid = RegularLatLonGrid(
    lat_start=29.5,
    lat_steps=657,
    lat_step_size=0.0625,
    lon_start=-23.5,
    lon_steps=1377,
    lon_step_size=0.0625
)

# DWD ICON D2 is regularized during download to nx: 1215, ny: 746 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Icon/Icon.swift#L150
_dwd_icon_d2_grid = RegularLatLonGrid(
    lat_start=43.18,
    lat_steps=746,
    lat_step_size=0.02,
    lon_start=-3.94,
    lon_steps=1215,
    lon_step_size=0.02
)

# DWD ICON EPS global is regularized during download to nx: 1439, ny: 721 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Icon/Icon.swift#L153
_dwd_icon_eps_grid = RegularLatLonGrid(
    lat_start=-90,
    lat_steps=721,
    lat_step_size=0.25,
    lon_start=-180,
    lon_steps=1439,
    lon_step_size=0.25
)

# DWD ICON EU EPS is regularized during download to nx: 689, ny: 329 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Icon/Icon.swift#L156
_dwd_icon_eu_eps_grid = RegularLatLonGrid(
    lat_start=29.5,
    lat_steps=329,
    lat_step_size=0.125,
    lon_start=-23.5,
    lon_steps=689,
    lon_step_size=0.125
)

# DWD ICON D2 EPS is regularized during download to nx: 1214, ny: 745 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Icon/Icon.swift#L160
_dwd_icon_d2_eps_grid = RegularLatLonGrid(
    lat_start=43.18,
    lat_steps=745,
    lat_step_size=0.02,
    lon_start=-3.94,
    lon_steps=1214,
    lon_step_size=0.02
)

# ECMWF IFS grid is a regular global lat/lon grid, nx: 1440, ny: 721 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Ecmwf/EcmwfDomain.swift#L105
_ecmwf_ifs025_grid = RegularLatLonGrid(
    lat_start=-90,
    lat_steps=721,
    lat_step_size=360/1440,
    lon_start=-180,
    lon_steps=1440,
    lon_step_size=180/(721-1)
)

# Méteo-France ARPEGE Europe grid: nx: 741, ny: 521 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/MeteoFrance/MeteoFranceDomain.swift#L341
_meteofrance_arpege_europe_grid = RegularLatLonGrid(
    lat_start=20,
    lat_steps=521,
    lat_step_size=0.1,
    lon_start=-32,
    lon_steps=741,
    lon_step_size=0.1
)

# Méteo-France ARPEGE World grid: nx: 1440, ny: 721 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/MeteoFrance/MeteoFranceDomain.swift#L343
_meteofrance_arpege_world025_grid = RegularLatLonGrid(
    lat_start=-90,
    lat_steps=721,
    lat_step_size=0.25,
    lon_start=-180,
    lon_steps=1440,
    lon_step_size=0.25
)

# Méteo-France AROME France grid: nx: 1121, ny: 717 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/MeteoFrance/MeteoFranceDomain.swift#L345
_meteofrance_arome_france0025_grid = RegularLatLonGrid(
    lat_start=37.5,
    lat_steps=717,
    lat_step_size=0.025,
    lon_start=-12.0,
    lon_steps=1121,
    lon_step_size=0.025
)

# Méteo-France AROME France HD grid: nx: 2801, ny: 1791 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/MeteoFrance/MeteoFranceDomain.swift#L347
_meteofrance_arome_france_hd_grid = RegularLatLonGrid(
    lat_start=37.5,
    lat_steps=1791,
    lat_step_size=0.01,
    lon_start=-12.0,
    lon_steps=2801,
    lon_step_size=0.01
)

# GEM Global grid: nx: 2400, ny: 1201 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Gem/GemDomain.swift#L139
_gem_global_grid = RegularLatLonGrid(
    lat_start=-90,
    lat_steps=1201,
    lat_step_size=0.15,
    lon_start=-180,
    lon_steps=2400,
    lon_step_size=0.15
)

# GEM Regional grid: Uses Stereographic projection
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Gem/GemDomain.swift#L141
_gem_regional_projection = StereographicProjection(
    latitude=90,
    longitude=249,
    radius=6371229
)
_gem_regional_grid = ProjectionGrid.from_bounds(
    nx=935,
    ny=824,
    lat_range=(18.14503, 45.405453),
    lon_range=(217.10745, 349.8256),
    projection=_gem_regional_projection
)

# GEM HRDPS Continental grid: Uses RotatedLatLon projection
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Gem/GemDomain.swift#L143
_gem_hrdps_projection = RotatedLatLonProjection(
    lat_origin=-36.0885,
    lon_origin=245.305
)
_gem_hrdps_grid = ProjectionGrid.from_bounds(
    nx=2540,
    ny=1290,
    lat_range=(39.626034, 47.876457),
    lon_range=(-133.62952, -40.708557),
    projection=_gem_hrdps_projection
)

# GEM Global Ensemble grid: nx: 720, ny: 361 points
# https://github.com/open-meteo/open-meteo/blob/522917b1d6e72a7e6b7d4ae7dfb49b0c556a6992/Sources/App/Gem/GemDomain.swift#L145
_gem_global_ensemble_grid = RegularLatLonGrid(
    lat_start=-90,
    lat_steps=361,
    lat_step_size=0.5,
    lon_start=-180,
    lon_steps=720,
    lon_step_size=0.5
)

DOMAINS: dict[str, OmDomain] = {
    'cmc_gem_gdps': OmDomain(
        name='cmc_gem_gdps',
        grid=_gem_global_grid,
        file_length=110,  # From GemDomain.omFileLength for gem_global case
        temporal_resolution_seconds=3600*3  # 3-hourly data
    ),
    'cmc_gem_rdps': OmDomain(
        name='cmc_gem_rdps',
        grid=_gem_regional_grid,
        file_length=78+36,  # From GemDomain.omFileLength for gem_regional case
        temporal_resolution_seconds=3600  # Hourly data
    ),
    'cmc_gem_hrdps': OmDomain(
        name='cmc_gem_hrdps',
        grid=_gem_hrdps_grid,
        file_length=48+36,  # From GemDomain.omFileLength for gem_hrdps_continental case
        temporal_resolution_seconds=3600  # Hourly data
    ),
    'cmc_gem_geps': OmDomain(
        name='cmc_gem_geps',
        grid=_gem_global_ensemble_grid,
        file_length=384//3+48//3,  # From GemDomain.omFileLength for gem_global_ensemble case
        temporal_resolution_seconds=3600*3  # 3-hourly data
    ),
    'dwd_icon': OmDomain(
        name='dwd_icon',
        grid=_dwd_icon_grid,
        file_length=180+1+3*24,  # From IconDomains.omFileLength for icon case
        temporal_resolution_seconds=3600
    ),
    'dwd_icon_eu': OmDomain(
        name='dwd_icon_eu',
        grid=_dwd_icon_eu_grid,
        file_length=120+1+3*24,  # From IconDomains.omFileLength for iconEu case
        temporal_resolution_seconds=3600
    ),
    'dwd_icon_d2': OmDomain(
        name='dwd_icon_d2',
        grid=_dwd_icon_d2_grid,
        file_length=48+1+3*24,  # From IconDomains.omFileLength for iconD2 case
        temporal_resolution_seconds=3600
    ),
    'dwd_icon_d2_15min': OmDomain(
        name='dwd_icon_d2_15min',
        grid=_dwd_icon_d2_grid,  # Uses same grid as dwd_icon_d2
        file_length=48*4+3*24,  # From IconDomains.omFileLength for iconD2_15min case
        temporal_resolution_seconds=3600//4  # 15 minutes = 3600/4
    ),
    'dwd_icon_eps': OmDomain(
        name='dwd_icon_eps',
        grid=_dwd_icon_eps_grid,
        file_length=180+1+3*24,  # Same as non-eps version
        temporal_resolution_seconds=3600
    ),
    'dwd_icon_eu_eps': OmDomain(
        name='dwd_icon_eu_eps',
        grid=_dwd_icon_eu_eps_grid,
        file_length=120+1+3*24,  # Same as non-eps version
        temporal_resolution_seconds=3600
    ),
    'dwd_icon_d2_eps': OmDomain(
        name='dwd_icon_d2_eps',
        grid=_dwd_icon_d2_eps_grid,
        file_length=48+1+3*24,  # Same as non-eps version
        temporal_resolution_seconds=3600
    ),
    'ecmwf_ifs025': OmDomain(
        name='ecmwf_ifs025',
        grid=_ecmwf_ifs025_grid,
        file_length=104,
        temporal_resolution_seconds=3600*3
    ),
    'meteofrance_arpege_europe': OmDomain(
        name='meteofrance_arpege_europe',
        grid=_meteofrance_arpege_europe_grid,
        file_length=114+3*24,  # From MeteoFranceDomain.omFileLength for arpege_europe case
        temporal_resolution_seconds=3600
    ),
    'meteofrance_arpege_world025': OmDomain(
        name='meteofrance_arpege_world025',
        grid=_meteofrance_arpege_world025_grid,
        file_length=114+4*24,  # From MeteoFranceDomain.omFileLength for arpege_world case
        temporal_resolution_seconds=3600
    ),
    'meteofrance_arome_france0025': OmDomain(
        name='meteofrance_arome_france0025',
        grid=_meteofrance_arome_france0025_grid,
        file_length=36+3*24,  # From MeteoFranceDomain.omFileLength for arome_france case
        temporal_resolution_seconds=3600
    ),
    'meteofrance_arome_france_hd': OmDomain(
        name='meteofrance_arome_france_hd',
        grid=_meteofrance_arome_france_hd_grid,
        file_length=36+3*24,  # From MeteoFranceDomain.omFileLength for arome_france_hd case
        temporal_resolution_seconds=3600
    ),
    'meteofrance_arome_france0025_15min': OmDomain(
        name='meteofrance_arome_france0025_15min',
        grid=_meteofrance_arome_france0025_grid,  # Using the same grid as non-15min version
        file_length=24*2,  # From MeteoFranceDomain.omFileLength for arome_france_15min case
        temporal_resolution_seconds=900
    ),
    'meteofrance_arome_france_hd_15min': OmDomain(
        name='meteofrance_arome_france_hd_15min',
        grid=_meteofrance_arome_france_hd_grid,  # Using the same grid as non-15min version
        file_length=24*2,  # From MeteoFranceDomain.omFileLength for arome_france_hd_15min case
        temporal_resolution_seconds=900
    ),
    'meteofrance_arpege_europe_probabilities': OmDomain(
        name='meteofrance_arpege_europe_probabilities',
        grid=_meteofrance_arpege_europe_grid,  # Using the same grid as non-probabilities version
        file_length=(102+4*24)//3,  # From MeteoFranceDomain.omFileLength for arpege_europe_probabilities case
        temporal_resolution_seconds=3600*3
    ),
    'meteofrance_arpege_world025_probabilities': OmDomain(
        name='meteofrance_arpege_world025_probabilities',
        grid=_meteofrance_arpege_world025_grid,  # Using the same grid as non-probabilities version
        file_length=(102+4*24)//3,  # From MeteoFranceDomain.omFileLength for arpege_world_probabilities case
        temporal_resolution_seconds=3600*3
    )
    # Additional domains can be added here
}

# Domain aliases to match the names in GemDomain.swift
DOMAINS['gem_global'] = DOMAINS['cmc_gem_gdps']
DOMAINS['gem_regional'] = DOMAINS['cmc_gem_rdps']
DOMAINS['gem_hrdps_continental'] = DOMAINS['cmc_gem_hrdps']
DOMAINS['gem_global_ensemble'] = DOMAINS['cmc_gem_geps']
