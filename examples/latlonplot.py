import datetime
import json
from enum import Enum
from typing import Any, List, Optional, Tuple

import fsspec
import matplotlib.axes as ma
import matplotlib.figure as mf
import matplotlib.pyplot as plt
import numpy as np
import omfiles
from matplotlib.colorbar import Colorbar
from matplotlib.contour import QuadContourSet
from matplotlib.widgets import Button


class LatLonViewer:
    """Interactive viewer for latitude-longitude data across multiple timestamps."""

    def __init__(self, s3_file: str, timerange: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        # Initialize member variables with proper types
        self.s3_file: str = s3_file
        self.current_timestamp: int = 0
        self.timestamps = timerange
        self.lat = lat
        self.lon = lon

        # Create the figure and axes first
        self.fig: mf.Figure
        self.ax: ma.Axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Add navigation buttons
        button_next_ax = plt.axes((0.81, 0.05, 0.1, 0.075))
        self.next_button = Button(button_next_ax, "Next")
        self.next_button.on_clicked(self._forward_to_next)
        button_prev_ax = plt.axes((0.7, 0.05, 0.1, 0.075))
        self.prev_button = Button(button_prev_ax, "Previous")
        self.prev_button.on_clicked(self._backward_to_prev)

        # Initialize other attributes that will be set later
        self.contour: Optional[QuadContourSet] = None
        self.colorbar: Optional[Colorbar] = None

        # Initialize data connection
        self._initialize_data_connection()

        # Initial plot
        self._update_plot()

    def _initialize_data_connection(self) -> None:
        """Initialize connection to data file."""
        # Set up fsspec with mmap cache for better performance
        fs = fsspec.filesystem(protocol="s3", anon=True)

        # Open the file with caching to improve performance for repeated access
        file_obj = fs.open(
            self.s3_file, mode="rb", cache_type="mmap", block_size=1024 * 1024, cache_options={"location": "cache"}
        )

        # Create OM file reader
        self.reader = omfiles.OmFilePyReader(file_obj)
        self.file_obj = file_obj  # Store for later cleanup

        # Get the shape of the data
        data_shape = self.reader.shape
        print(f"Data shape: {data_shape}")

        if len(data_shape) != 3:
            self.close()
            raise ValueError(f"Unsupported data shape {data_shape}")

        if data_shape[0] != len(self.lat) or data_shape[1] != len(self.lon):
            self.close()
            raise ValueError(
                f"Data shape {data_shape} does not match lat/lon grid shape ({len(self.lat)}, {len(self.lon)})"
            )

        # Store data shape and initialize coordinate arrays
        self.data_shape: List[int] = data_shape

    def _update_plot(self) -> None:
        """Update the plot with the current timestamp."""
        # Clear existing plots
        self.ax.clear()

        # Get data for the current timestamp
        print(f"Reading data for timestamp {self.current_timestamp}...")
        data = self.reader[:, :, self.current_timestamp : self.current_timestamp + 1]
        if data.ndim == 3:
            data = data[:, :, 0]  # Remove the time dimension

        # Create a mesh grid for plotting
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)

        # Plot the data
        self.contour = self.ax.contourf(lon_grid, lat_grid, data, cmap="RdBu_r", levels=20)

        # Add or update colorbar
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.contour, ax=self.ax)
            self.colorbar.set_label("Temperature (°C)")
        else:
            self.colorbar.update_normal(self.contour)

        # Add labels and title
        self.ax.set_xlabel("Longitude (°)")
        self.ax.set_ylabel("Latitude (°)")
        current_time = self.timestamps[self.current_timestamp]
        self.ax.set_title(f"Temperature at 2m - {current_time}")

        # Add grid lines
        self.ax.grid(linestyle="--", alpha=0.5)

        # Redraw the canvas
        self.fig.canvas.draw_idle()

    def _forward_to_next(self, event: Any = None) -> None:
        """Move to the next timestamp."""
        if self.current_timestamp < self.data_shape[2] - 1:
            self.current_timestamp += 1
            self._update_plot()
            print(f"Moved to timestamp {self.current_timestamp}")
        else:
            print("Already at the last timestamp")

    def _backward_to_prev(self, event: Any = None) -> None:
        """Move to the previous timestamp."""
        if self.current_timestamp > 0:
            self.current_timestamp -= 1
            self._update_plot()
            print(f"Moved to timestamp {self.current_timestamp}")
        else:
            print("Already at the first timestamp")

    def show(self) -> None:
        """Display the interactive plot."""
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        plt.show()

    def close(self) -> None:
        """Close the reader and file objects."""
        # Use hasattr to check if attribute exists before accessing
        if hasattr(self, "reader") and self.reader is not None:
            self.reader.close()

        if hasattr(self, "file_obj") and self.file_obj is not None:
            self.file_obj.close()


class SupportedDomain(Enum):
    dwd_icon_d2 = "dwd_icon_d2"
    ecmwf_ifs025 = "ecmwf_ifs025"

    def file_length(self):
        if self == SupportedDomain.dwd_icon_d2:
            return 121
        elif self == SupportedDomain.ecmwf_ifs025:
            return 104
        else:
            raise ValueError(f"Unsupported domain {self}")

    def lat_lon_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self == SupportedDomain.dwd_icon_d2:
            # DWD ICON D2 is regularized during download to nx: 1215, ny: 721 points
            # https://github.com/open-meteo/open-meteo/blob/1753ebb4966d05f61b17dd5bdf59700788d4a913/Sources/App/Icon/Icon.swift#L154
            lat_start = 43.18
            lat_step_size = 0.02
            lat_steps = 746

            lon_start = -3.94
            lon_step_size = 0.02
            lon_steps = 1215

            lat = np.linspace(lat_start, lat_start + lat_step_size * lat_steps, lat_steps, endpoint=False)
            lon = np.linspace(lon_start, lon_start + lon_step_size * lon_steps, lon_steps, endpoint=False)
            return lat, lon
        elif self == SupportedDomain.ecmwf_ifs025:
            # ECMWF IFS grid is a regular global lat/lon grid
            # nx: 1440, ny: 721 points https://github.com/open-meteo/open-meteo/blob/1753ebb4966d05f61b17dd5bdf59700788d4a913/Sources/App/Ecmwf/EcmwfDomain.swift#L107
            lat = np.linspace(-90, 90, 721, endpoint=False)  # From -90 to 90 degrees with 721 points
            lon = np.linspace(-180, 180, 1440, endpoint=False)  # From -180 to 180 degrees with 1440 points
            return lat, lon
        else:
            raise ValueError(f"Unsupported domain {self}")


class SupportedVariable(Enum):
    temperature_2m = "temperature_2m"


def find_chunk_for_timestamp(target_time: datetime.datetime, domain: SupportedDomain) -> Tuple[int, np.ndarray]:
    """
    Find the chunk number that contains a specific timestamp.

    Args:
        target_time: The timestamp to find
        domain: The domain to search in

    Returns:
        Tuple containing the chunk number and the time range of the chunk
    """
    meta_file = f"openmeteo/data/{domain.value}/static/meta.json"
    # Load metadata from S3
    fs = fsspec.filesystem(protocol="s3", anon=True)
    with fs.open(meta_file, mode="r") as f:
        metadata = json.load(f)

    # Get domain-specific parameters, unfortunately some of them are currently
    # hardcoded in the open-meteo source code, others can be retrieved from the metadata
    dt_seconds = metadata["temporal_resolution_seconds"]
    om_file_length = domain.file_length()

    # Calculate seconds since epoch for the target time
    epoch = datetime.datetime(1970, 1, 1)
    target_seconds = int((target_time - epoch).total_seconds())

    # Calculate the chunk number
    chunk = target_seconds // (om_file_length * dt_seconds)

    # Calculate the timerange for the chunk as np.ndarray of datetime.datetime
    chunk_start = np.datetime64(epoch + datetime.timedelta(0, chunk * om_file_length * dt_seconds))
    chunk_end = np.datetime64(epoch + datetime.timedelta(0, (chunk + 1) * om_file_length * dt_seconds))
    print(f"Chunk {chunk} covers the timerange from {chunk_start} to {chunk_end}")
    dt_range = np.arange(
        chunk_start, chunk_end, np.timedelta64(datetime.timedelta(0, dt_seconds)), dtype="datetime64[s]"
    )

    return chunk, dt_range


def view_latlon_data_interactive(
    domain: SupportedDomain,
    variable: SupportedVariable,
    timestamp: datetime.datetime,
) -> None:
    """
    Create an interactive viewer for latitude-longitude data for a specific domain and variable.
    """

    chunk_name, timerange = find_chunk_for_timestamp(timestamp, domain)

    lat, lon = domain.lat_lon_grid()

    print(f"Using chunk {chunk_name} for timestamp {timestamp}")

    s3_file: str = f"openmeteo/data/{domain.value}/{variable.value}/chunk_{chunk_name}.om"
    viewer = LatLonViewer(s3_file, timerange, lat, lon)
    try:
        viewer.show()
    finally:
        viewer.close()


if __name__ == "__main__":
    view_latlon_data_interactive(
        domain=SupportedDomain.dwd_icon_d2, variable=SupportedVariable.temperature_2m, timestamp=datetime.datetime.now()
    )
