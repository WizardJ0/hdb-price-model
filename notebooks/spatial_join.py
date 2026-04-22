import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def load_geojson_files():
    """Load the three master plan GeoJSON files"""
    print("Loading GeoJSON files...")
    
    # 2014 Master Plan (for transactions ≤ 2016)
    gdf_2014 = gpd.read_file('../geojson/MasterPlan2014LandUse.geojson')
    
    # 2019 Master Plan (for transactions 2017-2022) 
    gdf_2019 = gpd.read_file('../geojson/AmendmenttoMasterPlan2019LandUselayer.geojson')
    
    # 2025 Master Plan (for test set - future transactions)
    gdf_2025 = gpd.read_file('../geojson/MasterPlan2025LandUseLayer.geojson')
    
    print(f"2014 Master Plan: {len(gdf_2014)} polygons")
    print(f"2019 Master Plan: {len(gdf_2019)} polygons") 
    print(f"2025 Master Plan: {len(gdf_2025)} polygons")
    
    return gdf_2014, gdf_2019, gdf_2025

def create_spatial_join(df, gdf_2014, gdf_2019, gdf_2025):
    """Add spatial features by joining with appropriate master plan"""
    
    # Create geometry column from lat/lon
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Ensure master plan files have same CRS
    gdf_2014 = gdf_2014.to_crs("EPSG:4326")
    gdf_2019 = gdf_2019.to_crs("EPSG:4326") 
    gdf_2025 = gdf_2025.to_crs("EPSG:4326")
    
    results = []
    
    for idx, row in gdf_points.iterrows():
        year = row['Tranc_Year']
        
        # Select appropriate master plan based on transaction year
        if year <= 2016:
            master_plan = gdf_2014
        elif year <= 2022:
            master_plan = gdf_2019
        else:
            master_plan = gdf_2025
            
        # Find which polygon contains this point
        try:
            containing_polygons = master_plan[master_plan.geometry.contains(row.geometry)]
            
            if len(containing_polygons) > 0:
                # Take the first match (should be unique)
                polygon = containing_polygons.iloc[0]
                
                # Extract land use and GPR
                lu_desc = polygon.get('LU_DESC', polygon.get('PLN_AREA_N', 'UNKNOWN'))
                gpr = polygon.get('GPR', polygon.get('GROSS_PLOT_RATIO', 0))
                
                results.append({
                    'LU_DESC': str(lu_desc),
                    'GPR': float(gpr) if pd.notna(gpr) else 0.0
                })
            else:
                results.append({
                    'LU_DESC': 'UNKNOWN',
                    'GPR': 0.0
                })
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append({
                'LU_DESC': 'UNKNOWN', 
                'GPR': 0.0
            })
    
    # Convert results to DataFrame and merge back
    spatial_df = pd.DataFrame(results)
    df_with_spatial = pd.concat([df.reset_index(drop=True), spatial_df], axis=1)
    
    return df_with_spatial

if __name__ == "__main__":
    # Test the spatial join
    print("Testing spatial join functionality...")
    
    # Load a small sample of train data
    train_sample = pd.read_csv('../data/train.csv', nrows=100)
    
    # Load GeoJSON files
    gdf_2014, gdf_2019, gdf_2025 = load_geojson_files()
    
    # Test spatial join
    train_with_spatial = create_spatial_join(train_sample, gdf_2014, gdf_2019, gdf_2025)
    
    print("Sample spatial features:")
    print(train_with_spatial[['Tranc_Year', 'LU_DESC', 'GPR']].head())
    print(f"Unique land uses: {train_with_spatial['LU_DESC'].nunique()}")
