import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

tmin_path = "TMIN DATASET"
tmax_path = "TMAX DATASET"

tmin_ds = xr.open_dataset(tmin_path)
tmax_ds = xr.open_dataset(tmax_path)
#print("Original latitudes:", tmin_ds['lat'].values)

# Define the bounding box for Washington state
lon_min, lon_max = -124.736342, -116.945392
lat_min, lat_max = 45.521208, 49.382808

# Slice the data for Washington state
tmax_subset = tmax_ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))
tmin_subset = tmin_ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

# Convert temperatures from Kelvin to Celsius
tmax_subset = tmax_subset['tmax'] - 273.15
tmin_subset = tmin_subset['tmin'] - 273.15

# Calculate GDD
def calc_gdd(tmin, tmax, base_temp=0):
    avg_temp = (tmin + tmax) / 2
    gdd = avg_temp - base_temp
    return np.maximum(gdd, 0)  # Set all negative values to 0

gdd_subset = calc_gdd(tmin_subset, tmax_subset)

# Filter data for January to April, and 1991 to 2020
gdd_filtered = gdd_subset.sel(day=slice('1991-01-01', '2020-04-30'))
gdd_filtered = gdd_filtered.where(gdd_filtered['day.month'] >= 1).where(gdd_filtered['day.month'] <= 4)

# Calculate seasonal GDD values
seasonal_gdd = gdd_filtered.groupby('day.year').sum(dim='day')

# Calculate the average seasonal GDD values
average_seasonal_gdd = seasonal_gdd.mean(dim='year')

# Create a figure and a GeoAxes object
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the extent of the map
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add the coastline, state borders, and ocean features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.OCEAN)

# Add the GDD values to the plot
lons = average_seasonal_gdd['lon']
lats = average_seasonal_gdd['lat']
gdd_data = average_seasonal_gdd.data
cmap = plt.get_cmap('Spectral_r')
cmap.set_under('lightgray')
print("tmax_subset shape:", tmax_subset.shape)
print("tmin_subset shape:", tmin_subset.shape)
print("gdd_subset shape:", gdd_subset.shape)
print("gdd_filtered shape:", gdd_filtered.shape)
print("seasonal_gdd shape:", seasonal_gdd.shape)
print("average_seasonal_gdd shape:", average_seasonal_gdd.shape)

vmin, vmax = 0, np.nanmax(gdd_data)
levels = np.arange(np.floor(vmin / 25) * 25, np.ceil(vmax / 25) * 25 + 1, 25)

gdd_data_masked = np.ma.masked_where(gdd_data <= 0, gdd_data)

contour = plt.contourf(lons, lats, gdd_data_masked, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
ax.annotate('Data Source: Gridmet', xy=(0.01, 0.03), xycoords='axes fraction', fontsize=10, color='k', backgroundcolor='w')

# Add colorbar
colorbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=0.5)
colorbar.set_label('Growing Degree Days (GDD)', fontsize=12)

# Set up gridlines and labels
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = ticker.MaxNLocator(8)
gl.ylocator = ticker.MaxNLocator(6)
gl.xlabel_style = {'size': 10, 'color': 'gray'}
gl.ylabel_style = {'size': 10, 'color': 'gray'}

# Set title
plt.title('Growing Degree Day Accumulation Jan-April \n(1991-2020) Washington State at 0°C', fontsize=14, fontweight='bold', pad=20)
ax.text(0.5, 1.01, 'Bud Break & Flowering', horizontalalignment='center', fontsize=10, transform=ax.transAxes)

# Save the figure
plt.savefig('Washington_GDD_Map_Jan_Apr.png', dpi=300, bbox_inches='tight')

# Show the figure
plt.show()

