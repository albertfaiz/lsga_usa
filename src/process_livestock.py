#!/usr/bin/env python
"""
Process Livestock GeoTIFFs and County Shapefile Data

This script creates a grid of points within each county (from the county shapefile),
samples the given GeoTIFF files for various livestock data, computes average values per county,
and saves the results to a CSV file.
"""

import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd
from shapely.geometry import Point, shape
import rasterio.warp

# ------------------------------------------------------------------------------
# 1. Function to Create a Grid of Points Within a Polygon
# ------------------------------------------------------------------------------
def create_grid_points_within_polygon(polygon, spacing):
    """
    Create a grid of points inside the provided polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)
    points = []
    for x in x_coords:
        for y in y_coords:
            p = Point(x, y)
            if polygon.contains(p):
                points.append((x, y))
    return points

# ------------------------------------------------------------------------------
# 2. Function to Compute the Average Raster Value Over a County
# ------------------------------------------------------------------------------
def average_raster_over_county(tif_file, county_polygon, county_crs, spacing=None, is_density=False):
    """
    For a given GeoTIFF file and county polygon, sample the raster at grid points within
    the polygon and return the average value. If the data represent density (e.g., 2020),
    convert the values to counts using an approximate pixel area conversion.
    """
    with rasterio.open(tif_file) as src:
        raster_crs = src.crs
        # Reproject county polygon if needed.
        if county_crs != raster_crs:
            geom_dict = county_polygon.__geo_interface__
            county_geom_reproj = rasterio.warp.transform_geom(
                county_crs.to_string(), raster_crs.to_string(), geom_dict
            )
            county_polygon = shape(county_geom_reproj)

        # Use the raster’s pixel resolution if spacing not provided.
        if spacing is None:
            spacing = src.res[0]

        points = create_grid_points_within_polygon(county_polygon, spacing)
        if len(points) == 0:
            return np.nan

        sampled_vals = [val[0] for val in src.sample(points)]
        sampled_vals = np.array(sampled_vals)
        nodata = src.nodata
        if nodata is not None:
            sampled_vals = sampled_vals[sampled_vals != nodata]
        if len(sampled_vals) == 0:
            return np.nan

        if is_density:
            transform = src.transform
            pixel_width = abs(transform[0])
            pixel_height = abs(transform[4])
            # Approximate conversion: 1 degree ≈ 111 km.
            pixel_area_sqkm = (pixel_width * 111) * (pixel_height * 111)
            sampled_vals = sampled_vals * pixel_area_sqkm

        return np.nanmean(sampled_vals)

# ------------------------------------------------------------------------------
# 3. Define the Livestock GeoTIFF Files (Relative Paths)
# ------------------------------------------------------------------------------
# Use a relative path so the script works both locally and on HPC.
data_dir = "./data"
tif_files = {
    'Buffalo': {
        2010: f"{data_dir}/5_Bf_2010_Da.tif",
        2015: f"{data_dir}/5_Bf_2015_Da.tif",
        2020: f"{data_dir}/GLW4-2020.D-DA.BFL.tif"
    },
    'Chicken': {
        2010: f"{data_dir}/5_Ch_2010_Da.tif",
        2015: f"{data_dir}/5_Ch_2015_Da.tif",
        2020: f"{data_dir}/GLW4-2020.D-DA.CHK.tif"
    },
    'Cattle': {
        2010: f"{data_dir}/5_Ct_2010_Da.tif",
        2015: f"{data_dir}/5_Ct_2015_Da.tif",
        2020: f"{data_dir}/GLW4-2020.D-DA.CTL.tif"
    },
    'Duck': {
        2010: f"{data_dir}/5_Dk_2010_Da.tif",
        2015: f"{data_dir}/5_Dk_2015_Da.tif",
        2020: None  # No 2020 data for Duck.
    },
    'Goat': {
        2010: f"{data_dir}/5_Gt_2010_Da.tif",
        2015: f"{data_dir}/5_Gt_2015_Da.tif",
        2020: f"{data_dir}/GLW4-2020.D-DA.GTS.tif"
    },
    'Horse': {
        2010: f"{data_dir}/5_Ho_2010_Da.tif",
        2015: f"{data_dir}/5_Ho_2015_Da.tif",
        2020: None  # No 2020 data for Horse.
    },
    'Pig': {
        2010: f"{data_dir}/5_Pg_2010_Da.tif",
        2015: f"{data_dir}/5_Pg_2015_Da.tif",
        2020: f"{data_dir}/GLW4-2020.D-DA.PGS.tif"
    },
    'Sheep': {
        2010: f"{data_dir}/5_Sh_2010_Da.tif",
        2015: f"{data_dir}/5_Sh_2015_Da.tif",
        2020: f"{data_dir}/GLW4-2020.D-DA.SHP.tif"
    }
}

# ------------------------------------------------------------------------------
# 4. Read the County Shapefile and Prepare the DataFrame
# ------------------------------------------------------------------------------
county_shapefile = f"{data_dir}/cb_2023_us_county_500k.shp"
counties = gpd.read_file(county_shapefile)
df_counties = counties[['GEOID', 'NAME']].copy()
counties_crs = counties.crs

# ------------------------------------------------------------------------------
# 5. Process Each Livestock Type and Year, Compute County Averages
# ------------------------------------------------------------------------------
for livestock_type, years_dict in tif_files.items():
    for year, tif_path in years_dict.items():
        col_name = f"{livestock_type.lower()}_{year}"
        if tif_path is None:
            df_counties[col_name] = np.nan
            print(f"Skipping {livestock_type} for {year} (no data).")
            continue
        print(f"Processing {livestock_type} for {year}...")
        df_counties[col_name] = counties.geometry.apply(
            lambda geom: average_raster_over_county(
                tif_path, geom, counties_crs, spacing=None, is_density=(year == 2020)
            )
        )

# ------------------------------------------------------------------------------
# 6. Save the Final County-Level Averages to CSV
# ------------------------------------------------------------------------------
output_csv = "./county_average_livestock.csv"
df_counties.to_csv(output_csv, index=False)
print(f"\nCounty-level averages saved to {output_csv}")
