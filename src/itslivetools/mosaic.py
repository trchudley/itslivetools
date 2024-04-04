"""Functions for downloading ITS_LIVE annual velocity mosaics."""

import json, re, warnings

import xarray as xr
import rioxarray as rxr
import geopandas as gpd

from shapely.geometry import box
from pandas import Series

from urllib.request import urlopen
from typing import Tuple, List


VALID_LOCATIONS = {
    "greenland": "RGI05A",
    "antarctica": "RGI19A",
}

LOCATION_EPSG = {
    "greenland": 3413,
    "antarctica": 3031,
}

VARIABLES_ALL = [
    "count",
    "count0",
    "dt_max",
    "dv_dt",
    "dvx_dt",
    "dvy_dt",
    "floatingice",
    "landice",
    "mapping",
    "outlier_percent",
    "sensor_flag",
    "v",
    "v0",
    "v0_error",
    "v_amp",
    "v_amp_error",
    "v_error",
    "v_phase",
    "vx",
    "vx0",
    "vx0_error",
    "vx_amp",
    "vx_amp_error",
    "vx_error",
    "vx_phase",
    "vy",
    "vy0",
    "vy0_error",
    "vy_amp",
    "vy_amp_error",
    "vy_error",
    "vy_phase",
]

VARIABLES_DEFAULT = [
    "mapping",
    "v",
    "v_error",
    "vx",
    "vx_error",
    "vy",
    "vy_error",
]


def get_tiles(
    bounds: Tuple[float, float, float, float],  # must be in relevant EPSG
    region: str = "greenland",
) -> gpd.GeoDataFrame:
    """Returns a GeoPandas DataFrame containing the ITS_LIVE mosaic tiles that intersect
    the given bounding box. Bounding box must be in the same EPSG as the region of
    interest (e.g. EPSG:3413 for Greenland). Currently only accepts Greenland and
    Antarctica.

    Args:
        bounds (Tuple[float, float, float, float]): Bounding box of region of interest,
            in the form (xmin, ymin, xmax, ymax) and the same EPSG as the region of
            interest.
        region (str, optional): Region of interest. Defaults to "greenland". Currently
            only accepts "greenland" and "antarctica".

    Returns:
        gpd.GeoDataFrame: GeoPandas DataFrame containing the ITS_LIVE mosaic tiles that
            intersect the given bounding box.
    """

    # NOTE TO SELF: RETURNED TILES COULD BE USED FOR MOSAICS OR PAIRS

    # Mosaics are located at:
    # https://its-live-data.s3.amazonaws.com/index.html#mosaics/annual/v2/
    # Single-EPSG regions appear to be in /netcdf, and multi-EPSG in /originalMultiEPSG
    # Focus on /netcdf for now

    # Check region is valid
    region = region.lower()
    if region not in VALID_LOCATIONS.keys():
        raise ValueError(f"Region must be one of {VALID_LOCATIONS.keys()}")
    rgi_id = VALID_LOCATIONS[region]
    region_epsg = LOCATION_EPSG[region]

    # Load json
    cube_json_url = f"https://its-live-data.s3-us-west-2.amazonaws.com/mosaics/annual/v2/netcdf/ITS_LIVE_velocity_120m_{rgi_id}_0000_v02.json"
    with urlopen(cube_json_url) as url:
        cube_json = json.load(url)

    # Remove components of json that are not of full length (assumed first entry is)
    length = None
    cube_dict = {}

    for key in cube_json.keys():
        if not length:
            length = len(cube_json[key])
        if len(cube_json[key]) == length:
            cube_dict[key] = cube_json[key]
        else:
            pass
            # print(key)
            # print(cube_json[key])

    # Construct geometries for geopandas dataframe, getting X and Y centroids from
    # name of zarr on S3
    geometries = []
    epsg_list = []
    for string in cube_dict["composites_s3"]:

        x_pattern = r"(?<=_X)(.*?)(?=_)"
        x_matches = re.findall(x_pattern, string)

        y_pattern = r"(?<=_Y)(.*?)(?=.zarr)"
        y_matches = re.findall(y_pattern, string)

        epsg_pattern = r"(?<=_EPSG)(.*?)(?=_)"
        epsg_matches = re.findall(epsg_pattern, string)

        try:
            x = int(x_matches[0])
            y = int(y_matches[0])
            epsg_matches = int(epsg_matches[0])
        except:
            print("X:", x_matches)
            print("Y:", y_matches)
            print("EPSG:", epsg_matches)
            raise ValueError("Failed to find X/Y coordinates or EPSG")

        # Cubes have a 100 km diameter, so must buffer points by 50 km
        r = 50000
        geom = box(x - r, y - r, x + r, y + r)
        geometries.append(geom)

        epsg_list.append(epsg_matches)

    # Ensure individual list
    epsg_list = list(set(epsg_list))
    if len(epsg_list) != 1:
        raise ValueError(
            "All cubes must have same EPSG. Multiple EPSG region handling not yet implemented."
        )
    else:
        epsg = epsg_list[0]

    if epsg != region_epsg:
        raise ValueError(
            f"Reported EPSG code from cube does not match that expected. Received {epsg}, expected {region_epsg}."
        )

    # Create geopandas dataframe
    gdf = gpd.GeoDataFrame(cube_dict, geometry=geometries, crs=epsg)
    gdf = gdf[gdf.intersects(box(*bounds))]

    return gdf


def download_tile(
    s3_location: str | gpd.GeoDataFrame | Series,
    bounds: Tuple[float, float, float, float] = None,
    year: int | Tuple[int, int] = None,
    variables: List[str] = VARIABLES_DEFAULT,
) -> xr.Dataset:
    """Downloads a single file from the ITS_LIVE AWS bucket, based on the s3 url.

    Args:
        s3_location (str): Location of s3 zarr file. Can be a string extracted from the
            tile GeoDataFrame, a row of the GeoDataFrame, or the GeoDataFrame itself
            (in this case, only the first tile will be downloaded).
        bounds (Tuple[float, float, float, float]): Bounds of tile (xmin, ymin, xmax,
            ymax) in matching EPSG to tile.
        year (int | Tuple[int, int]): Years to download. Can be single year or range of
            years as (year_min, year_max).
        variables (List[str], optional): Variables to download. All availablle variables
            can be viewed using itslivetools.mosaic.VARIABLES_ALL. Defaults to
            itslivetools.mosaic.VARIABLES_DEFAULT.

    Returns:
        xr.Dataset: Xarray Dataset containing the requested annual mosaics.
    """

    # Extract s3_url from s3_location
    if type(s3_location) == gpd.GeoDataFrame:
        if len(s3_location) > 1:
            warnings.warn(
                "Received multi-polygon GeoDataFrame. Only one tile can be downloaded at a time. Downloading first tile."
            )
        s3_url = s3_location.composites_s3.values[0]
    elif type(s3_location) == Series:
        s3_url = s3_location.composites_s3
    elif type(s3_location) == str:
        s3_url = s3_location
    else:
        raise ValueError(
            "`s3_location` must be GeoDataFrame, GeoDataFrame row, or string"
        )

    # Sanitise year input, if it is specified
    if year:
        if type(year) == int:
            year_start = year
            year_end = year
        elif (type(year) == list) or (type(year) == tuple):
            if not len(year) == 2:
                raise ValueError("`year` list must be length == 2")
            year_start = year[0]
            year_end = year[1]
        else:
            raise ValueError("`year` must be integer or list of two integers")

    # Begin downloading...

    # Open dataset
    xds = xr.open_dataset(
        s3_url, engine="zarr", chunks="auto", storage_options={"anon": True}
    )

    # filter to variables
    xds = xds[variables]
    xds = xds.rio.write_crs(int(xds.attrs["projection"]))

    # filter based on year, if it is specified
    if year:
        xds = xds.where(
            (xds.time.dt.year >= year_start) & (xds.time.dt.year <= year_end), drop=True
        )

    # filter to region, if the bounding box is specified
    if bounds:
        xds = xds.rio.clip_box(*bounds)
        xds = xds.rio.pad_box(*bounds)

    return xds


def merge_tiles(xds_list: List[xr.Dataset]) -> xr.Dataset:
    """Merges different ITS_LIVE tiles into one, using xarray merge function.

    Args:
        xds_list (List[xr.Dataset]): List of tiles to merge.

    Returns:
        xr.Dataset: Merged dataset.
    """

    return xr.merge(xds_list, combine_attrs="drop_conflicts")
