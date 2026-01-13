"""Domains used in Open-Meteo files."""

from typing import List

import numpy as np

from omfiles._utils import EPOCH


class OmDomain:
    """
    Class representing a domain configuration for a weather model.
    """

    def __init__(self, name: str, file_length: int, temporal_resolution_seconds: int = 3600):
        """
        Initialize a domain configuration.

        Args:
            name (str): Name of the domain.
            file_length (int): Number of time steps in each file chunk.
            temporal_resolution_seconds (int, optional): Time resolution in seconds. Defaults to 3600 (1 hour).
        """
        self.name = name
        self.file_length = file_length
        self.temporal_resolution_seconds = temporal_resolution_seconds

    def time_to_chunk_index(self, timestamp: np.datetime64) -> int:
        """
        Convert a timestamp to a chunk index.

        This depends on the file_length and the temporal_resolution_seconds of the domain.

        Args:
            timestamp (np.datetime64): The timestamp to convert.

        Returns:
            int: The chunk index containing the timestamp.
        """
        seconds_since_epoch = (timestamp - EPOCH) / np.timedelta64(1, "s")
        chunk_index = int(seconds_since_epoch / (self.file_length * self.temporal_resolution_seconds))
        return chunk_index

    def chunks_for_date_range(
        self,
        start_timestamp: np.datetime64,
        end_timestamp: np.datetime64,
    ) -> List[int]:
        """
        Find all chunk indices that contain data within the given date range.

        Args:
            start_timestamp (np.datetime64): Start timestamp for the data range.
            end_timestamp (np.datetime64): End timestamp for the data range.

        Returns:
            List[int]: List of chunk indices containing data within the date range.
        """
        # Get chunk indices for start and end dates
        start_chunk = self.time_to_chunk_index(start_timestamp)
        end_chunk = self.time_to_chunk_index(end_timestamp)

        # Generate list of all chunks between start and end (inclusive)
        return list(range(start_chunk, end_chunk + 1))

    def get_chunk_time_range(self, chunk_index: int):
        """
        Get the time range covered by a specific chunk.

        Args:
            chunk_index (int): Index of the chunk.

        Returns:
            np.ndarray: Array of datetime64 objects representing the time points in the chunk.
        """
        chunk_start_seconds = chunk_index * self.file_length * self.temporal_resolution_seconds
        start_time = EPOCH + np.timedelta64(chunk_start_seconds, "s")

        # Generate timestamps at regular intervals from the start time
        time_delta = np.timedelta64(self.temporal_resolution_seconds, "s")
        # Note: better type inference via list comprehension here
        timestamps = np.array([start_time + i * time_delta for i in range(self.file_length)])
        return timestamps


DOMAINS: dict[str, OmDomain] = {
    "cmc_gem_gdps": OmDomain(
        name="cmc_gem_gdps",
        file_length=110,  # From GemDomain.omFileLength for gem_global case
        temporal_resolution_seconds=3600 * 3,  # 3-hourly data
    ),
    "cmc_gem_rdps": OmDomain(
        name="cmc_gem_rdps",
        file_length=78 + 36,  # From GemDomain.omFileLength for gem_regional case
        temporal_resolution_seconds=3600,  # Hourly data
    ),
    "cmc_gem_hrdps": OmDomain(
        name="cmc_gem_hrdps",
        file_length=48 + 36,  # From GemDomain.omFileLength for gem_hrdps_continental case
        temporal_resolution_seconds=3600,  # Hourly data
    ),
    "cmc_gem_geps": OmDomain(
        name="cmc_gem_geps",
        file_length=384 // 3 + 48 // 3,  # From GemDomain.omFileLength for gem_global_ensemble case
        temporal_resolution_seconds=3600 * 3,  # 3-hourly data
    ),
    "dwd_icon": OmDomain(
        name="dwd_icon",
        file_length=180 + 1 + 3 * 24,  # From IconDomains.omFileLength for icon case
        temporal_resolution_seconds=3600,
    ),
    "dwd_icon_eu": OmDomain(
        name="dwd_icon_eu",
        file_length=120 + 1 + 3 * 24,  # From IconDomains.omFileLength for iconEu case
        temporal_resolution_seconds=3600,
    ),
    "dwd_icon_d2": OmDomain(
        name="dwd_icon_d2",
        file_length=48 + 1 + 3 * 24,  # From IconDomains.omFileLength for iconD2 case
        temporal_resolution_seconds=3600,
    ),
    "dwd_icon_d2_15min": OmDomain(
        name="dwd_icon_d2_15min",
        file_length=48 * 4 + 3 * 24,  # From IconDomains.omFileLength for iconD2_15min case
        temporal_resolution_seconds=3600 // 4,  # 15 minutes = 3600/4
    ),
    "dwd_icon_eps": OmDomain(
        name="dwd_icon_eps",
        file_length=180 + 1 + 3 * 24,  # Same as non-eps version
        temporal_resolution_seconds=3600,
    ),
    "dwd_icon_eu_eps": OmDomain(
        name="dwd_icon_eu_eps",
        file_length=120 + 1 + 3 * 24,  # Same as non-eps version
        temporal_resolution_seconds=3600,
    ),
    "dwd_icon_d2_eps": OmDomain(
        name="dwd_icon_d2_eps",
        file_length=48 + 1 + 3 * 24,  # Same as non-eps version
        temporal_resolution_seconds=3600,
    ),
    "ecmwf_ifs025": OmDomain(name="ecmwf_ifs025", file_length=104, temporal_resolution_seconds=3600 * 3),
    "meteofrance_arpege_europe": OmDomain(
        name="meteofrance_arpege_europe",
        file_length=114 + 3 * 24,  # From MeteoFranceDomain.omFileLength for arpege_europe case
        temporal_resolution_seconds=3600,
    ),
    "meteofrance_arpege_world025": OmDomain(
        name="meteofrance_arpege_world025",
        file_length=114 + 4 * 24,  # From MeteoFranceDomain.omFileLength for arpege_world case
        temporal_resolution_seconds=3600,
    ),
    "meteofrance_arome_france0025": OmDomain(
        name="meteofrance_arome_france0025",
        file_length=36 + 3 * 24,  # From MeteoFranceDomain.omFileLength for arome_france case
        temporal_resolution_seconds=3600,
    ),
    "meteofrance_arome_france_hd": OmDomain(
        name="meteofrance_arome_france_hd",
        file_length=36 + 3 * 24,  # From MeteoFranceDomain.omFileLength for arome_france_hd case
        temporal_resolution_seconds=3600,
    ),
    "meteofrance_arome_france0025_15min": OmDomain(
        name="meteofrance_arome_france0025_15min",
        file_length=24 * 2,  # From MeteoFranceDomain.omFileLength for arome_france_15min case
        temporal_resolution_seconds=900,
    ),
    "meteofrance_arome_france_hd_15min": OmDomain(
        name="meteofrance_arome_france_hd_15min",
        file_length=24 * 2,  # From MeteoFranceDomain.omFileLength for arome_france_hd_15min case
        temporal_resolution_seconds=900,
    ),
    "meteofrance_arpege_europe_probabilities": OmDomain(
        name="meteofrance_arpege_europe_probabilities",
        file_length=(102 + 4 * 24) // 3,  # From MeteoFranceDomain.omFileLength for arpege_europe_probabilities case
        temporal_resolution_seconds=3600 * 3,
    ),
    "meteofrance_arpege_world025_probabilities": OmDomain(
        name="meteofrance_arpege_world025_probabilities",
        file_length=(102 + 4 * 24) // 3,  # From MeteoFranceDomain.omFileLength for arpege_world_probabilities case
        temporal_resolution_seconds=3600 * 3,
    ),
    # Additional domains can be added here
}

# Domain aliases to match the names in GemDomain.swift
DOMAINS["gem_global"] = DOMAINS["cmc_gem_gdps"]
DOMAINS["gem_regional"] = DOMAINS["cmc_gem_rdps"]
DOMAINS["gem_hrdps_continental"] = DOMAINS["cmc_gem_hrdps"]
DOMAINS["gem_global_ensemble"] = DOMAINS["cmc_gem_geps"]
