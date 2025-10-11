#!/usr/bin/env python
"""
USGS Earthquake Data Download Script
Downloads California earthquake data for deep learning project.

Data Source: USGS (United States Geological Survey) only
Total Download: ~70MB
Time: 5-10 minutes

Usage: python scripts/download_usgs_data.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project configuration
CONFIG = {
    'region': 'California',
    'min_magnitude': 3.5,
    'max_events': 20000,
    'start_date': '1990-01-01',
    'end_date': '2024-12-31',
    'output_dir': 'data/raw',
    
    # Geographic bounds for California
    'min_latitude': 32.0,
    'max_latitude': 42.0,
    'min_longitude': -125.0,
    'max_longitude': -114.0
}


def create_directories():
    """Create necessary directory structure"""
    dirs = [
        'data/raw/earthquakes',
        'data/raw/external/faults',
        'data/raw/external/plates',
        'data/processed',
        'notebooks',
        'models'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("✓ Directory structure created")


def download_usgs_earthquakes():
    """Download earthquake catalog from USGS"""
    logger.info("=" * 70)
    logger.info("STEP 1: DOWNLOADING USGS EARTHQUAKE CATALOG")
    logger.info("=" * 70)
    
    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime
        
        # Connect to USGS
        client = Client("USGS")
        logger.info("✓ Connected to USGS earthquake catalog")
        
        logger.info(f"\nDownload Parameters:")
        logger.info(f"  Region: {CONFIG['region']}")
        logger.info(f"  Latitude: {CONFIG['min_latitude']}° to {CONFIG['max_latitude']}°")
        logger.info(f"  Longitude: {CONFIG['min_longitude']}° to {CONFIG['max_longitude']}°")
        logger.info(f"  Date Range: {CONFIG['start_date']} to {CONFIG['end_date']}")
        logger.info(f"  Min Magnitude: {CONFIG['min_magnitude']}")
        logger.info(f"  Max Events: {CONFIG['max_events']}")
        
        # Download earthquake catalog
        logger.info("\nDownloading earthquake catalog...")
        catalog = client.get_events(
            starttime=UTCDateTime(CONFIG['start_date']),
            endtime=UTCDateTime(CONFIG['end_date']),
            minmagnitude=CONFIG['min_magnitude'],
            minlatitude=CONFIG['min_latitude'],
            maxlatitude=CONFIG['max_latitude'],
            minlongitude=CONFIG['min_longitude'],
            maxlongitude=CONFIG['max_longitude'],
            limit=CONFIG['max_events'],
            orderby='magnitude'
        )
        
        logger.info(f"✓ Downloaded {len(catalog)} earthquake events")
        
        # Convert to DataFrame with comprehensive features
        logger.info("\nProcessing earthquake data...")
        events = []
        
        for event in catalog:
            origin = event.preferred_origin()
            magnitude = event.preferred_magnitude()
            
            if origin and magnitude:
                time = origin.time.datetime
                
                event_data = {
                    # Basic identification
                    'event_id': str(event.resource_id).split('/')[-1],
                    
                    # Time information
                    'time': time,
                    'year': time.year,
                    'month': time.month,
                    'day': time.day,
                    'hour': time.hour,
                    'minute': time.minute,
                    'day_of_week': time.weekday(),
                    'day_of_year': time.timetuple().tm_yday,
                    'timestamp': pd.Timestamp(time).value // 10**9,
                    
                    # Location
                    'latitude': origin.latitude,
                    'longitude': origin.longitude,
                    'depth_km': origin.depth / 1000 if origin.depth else None,
                    
                    # Magnitude
                    'magnitude': magnitude.mag,
                    'magnitude_type': magnitude.magnitude_type,
                    
                    # Quality metrics (with safe attribute access)
                    'horizontal_uncertainty_km': getattr(origin, 'horizontal_uncertainty', None) / 1000 if getattr(origin, 'horizontal_uncertainty', None) else None,
                    'depth_uncertainty_km': getattr(origin, 'depth_uncertainty', None) / 1000 if getattr(origin, 'depth_uncertainty', None) else None,
                    'has_uncertainty': hasattr(origin, 'origin_uncertainty') and origin.origin_uncertainty is not None,
                    'time_uncertainty_sec': origin.time_errors.uncertainty if hasattr(origin, 'time_errors') and origin.time_errors else None,
                    'azimuthal_gap_deg': getattr(origin, 'azimuthal_gap', None),
                    'stations_used': origin.quality.used_station_count if hasattr(origin, 'quality') and origin.quality else None,
                    'min_station_distance_deg': origin.quality.minimum_distance if hasattr(origin, 'quality') and origin.quality else None,
                    
                    # Event description
                    'description': event.event_descriptions[0].text if event.event_descriptions else None
                }
                
                events.append(event_data)
        
        # Create DataFrame
        df = pd.DataFrame(events)
        
        # Add computed features
        logger.info("Computing additional features...")
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        # Save to CSV
        output_path = Path(CONFIG['output_dir']) / 'earthquakes' / 'california_earthquakes.csv'
        df.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Saved {len(df)} events to {output_path}")
        
        # Display statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"  Total Events: {len(df)}")
        logger.info(f"  Date Range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"  Magnitude Range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
        logger.info(f"  Mean Magnitude: {df['magnitude'].mean():.2f}")
        logger.info(f"  Depth Range: {df['depth_km'].min():.1f} - {df['depth_km'].max():.1f} km")
        logger.info(f"  Mean Depth: {df['depth_km'].mean():.1f} km")
        logger.info(f"  Years Covered: {df['year'].nunique()}")
        logger.info(f"  Magnitude Types: {', '.join(df['magnitude_type'].unique())}")
        
        return True, df
        
    except ImportError:
        logger.error("❌ ObsPy not installed")
        logger.error("Install with: pip install obspy")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error downloading USGS data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def download_fault_data():
    """Download USGS Quaternary fault data"""
    logger.info("=" * 70)
    logger.info("STEP 2: DOWNLOADING FAULT LINE DATA")
    logger.info("=" * 70)
    
    try:
        import urllib.request
        import zipfile
        
        fault_url = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip"
        fault_dir = Path(CONFIG['output_dir']) / 'external' / 'faults'
        fault_zip = fault_dir / 'faults.zip'
        
        logger.info("Downloading USGS Quaternary Fault Database...")
        logger.info(f"  Source: {fault_url}")
        
        os.makedirs(fault_dir, exist_ok=True)
        urllib.request.urlretrieve(fault_url, fault_zip)
        
        logger.info("Extracting fault data...")
        with zipfile.ZipFile(fault_zip, 'r') as zip_ref:
            zip_ref.extractall(fault_dir)
        
        # Remove zip file
        fault_zip.unlink()
        
        logger.info(f"✓ Fault data saved to: {fault_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading fault data: {e}")
        logger.info("  (This is optional - continuing without fault data)")
        return False


def download_plate_boundaries():
    """Download tectonic plate boundary data"""
    logger.info("=" * 70)
    logger.info("STEP 3: DOWNLOADING PLATE BOUNDARY DATA")
    logger.info("=" * 70)
    
    try:
        import urllib.request
        
        plate_url = "http://peterbird.name/oldftp/PB2002/PB2002_boundaries.dig.txt"
        plate_path = Path(CONFIG['output_dir']) / 'external' / 'plates' / 'plate_boundaries.txt'
        
        logger.info("Downloading PB2002 plate boundary model...")
        logger.info(f"  Source: {plate_url}")
        
        os.makedirs(plate_path.parent, exist_ok=True)
        urllib.request.urlretrieve(plate_url, plate_path)
        
        logger.info(f"✓ Plate boundaries saved to: {plate_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading plate data: {e}")
        logger.info("  (This is optional - continuing without plate data)")
        return False


def create_summary(earthquake_df, fault_success, plate_success):
    """Create comprehensive download summary"""
    logger.info("=" * 70)
    logger.info("CREATING DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    
    summary = {
        'download_info': {
            'date': datetime.now().isoformat(),
            'script_version': '1.0',
            'data_source': 'USGS (United States Geological Survey)'
        },
        'configuration': CONFIG,
        'download_results': {
            'earthquakes': len(earthquake_df) if earthquake_df is not None else 0,
            'faults': fault_success,
            'plates': plate_success
        }
    }
    
    if earthquake_df is not None:
        summary['statistics'] = {
            'total_events': len(earthquake_df),
            'date_range': {
                'start': str(earthquake_df['time'].min()),
                'end': str(earthquake_df['time'].max())
            },
            'magnitude': {
                'min': float(earthquake_df['magnitude'].min()),
                'max': float(earthquake_df['magnitude'].max()),
                'mean': float(earthquake_df['magnitude'].mean()),
                'median': float(earthquake_df['magnitude'].median())
            },
            'depth_km': {
                'min': float(earthquake_df['depth_km'].min()),
                'max': float(earthquake_df['depth_km'].max()),
                'mean': float(earthquake_df['depth_km'].mean())
            },
            'temporal': {
                'years_covered': int(earthquake_df['year'].nunique()),
                'events_per_year': float(len(earthquake_df) / earthquake_df['year'].nunique())
            }
        }
    
    # Save summary
    summary_path = Path(CONFIG['output_dir']) / 'download_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"✓ Summary saved to: {summary_path}")
    
    return summary


def main():
    """Main execution function"""
    
    print("\n" + "=" * 70)
    print("USGS EARTHQUAKE DATA DOWNLOAD")
    print("Deep Learning Project - California Seismic Data")
    print("=" * 70)
    
    print("\n📊 What You'll Get:")
    print(f"  ✓ {CONFIG['max_events']} California earthquakes (magnitude ≥ {CONFIG['min_magnitude']})")
    print(f"  ✓ Date range: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print(f"  ✓ Comprehensive metadata (location, time, magnitude, quality)")
    print(f"  ✓ Fault lines and plate boundaries (context data)")
    
    print("\n📦 Data Source:")
    print("  ✓ USGS - United States Geological Survey")
    print("  ✓ Official government earthquake catalog")
    print("  ✓ Gold standard for seismic research")
    
    print("\n💾 Download Size:")
    print("  ✓ Earthquake data: ~5-20 MB")
    print("  ✓ Fault data: ~30-40 MB")
    print("  ✓ Plate data: ~5 MB")
    print("  ✓ Total: ~50-70 MB")
    
    print("\n⏱️  Estimated Time: 5-10 minutes")

    print("\n" + "=" * 70)
    
    response = input("\n🚀 Start download? (y/n): ")
    if response.lower() != 'y':
        print("\n❌ Download cancelled")
        return
    
    print("\n")
    
    # Create directory structure
    create_directories()
    
    # Download data
    earthquake_success, earthquake_df = download_usgs_earthquakes()
    fault_success = download_fault_data()
    plate_success = download_plate_boundaries()
    
    # Create summary
    if earthquake_success:
        create_summary(earthquake_df, fault_success, plate_success)
    
    # Final report
    print("\n" + "=" * 70)
    
    if earthquake_success:
        print("✅ DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📁 Your Data:")
        print(f"  📄 {CONFIG['output_dir']}/earthquakes/california_earthquakes.csv")
        if fault_success:
            print(f"  📁 {CONFIG['output_dir']}/external/faults/")
        if plate_success:
            print(f"  📄 {CONFIG['output_dir']}/external/plates/plate_boundaries.txt")
        print(f"  📄 {CONFIG['output_dir']}/download_summary.json")
        
        print("\n🎯 Models You Can Build:")
        print("  1️⃣  Dense Neural Network")
        print("     - Magnitude prediction from location/time")
        print("     - Multi-feature regression")
        
        print("  2️⃣  LSTM/RNN")
        print("     - Temporal sequence modeling")
        print("     - Time-series earthquake patterns")
        
        print("  3️⃣  CNN")
        print("     - Spatial grid-based analysis")
        print("     - Geographic earthquake clustering")
        
        print("  4️⃣  Graph Neural Network (GNN)")
        print("     - Fault network modeling")
        print("     - Earthquake relationship graphs")
        
        print("\n📊 Next Steps:")
        print("  1. Explore your data:")
        print("     jupyter notebook notebooks/01_data_exploration.ipynb")
        
        print("\n  2. Feature engineering:")
        print("     - Temporal features (hour, day, season)")
        print("     - Spatial features (distance to faults)")
        print("     - Historical features (recent activity)")
        
        print("\n  3. Build and compare models:")
        print("     - Train all 4 architectures")
        print("     - Use same train/test split")
        print("     - Compare metrics (MSE, MAE, R²)")
        
        print("\n  4. Write your report:")
        print("     - Dataset description ✓")
        print("     - Model architectures")
        print("     - Results & comparison")
        print("     - Critical analysis")
        
        print("\n💡 Optional Enhancement:")
        print("  Later, you can add STEAD waveforms (Chunk 2)")
        print("  for advanced signal processing models")
        
    else:
        print("❌ DOWNLOAD FAILED")
        print("=" * 70)
        print("\n🔧 Troubleshooting:")
        print("  1. Install ObsPy: pip install obspy")
        print("  2. Check internet connection")
        print("  3. Verify USGS services are online")
        print("  4. Check error messages above")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()