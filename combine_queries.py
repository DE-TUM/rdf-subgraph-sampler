#!/usr/bin/env python3
"""
Script to combine multiple query JSON files into one single file.
"""

import json
import glob
import os
import sys
from datetime import datetime

def combine_query_files(pattern="*_stars_*.json", output_file=None):
    """
    Combine multiple query JSON files into one.
    
    Args:
        pattern: Glob pattern to match query files (can be absolute or relative path)
        output_file: Output filename (auto-generated if None)
    """
    
    # Expand the pattern to handle absolute paths and user home directory
    pattern = os.path.expanduser(pattern)
    
    # Find all matching files
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} files to combine:")
    for f in sorted(files):
        print(f"  - {f}")
    
    combined_queries = []
    total_files_processed = 0
    total_queries_filtered = 0
    
    for file_path in sorted(files):
        try:
            print(f"Processing {file_path}...")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Filter out queries with y value of -1
                original_count = len(data)
                filtered_data = [query for query in data if query.get('y') != -1]
                filtered_count = original_count - len(filtered_data)
                
                combined_queries.extend(filtered_data)
                total_queries_filtered += filtered_count
                
                print(f"  Added {len(filtered_data)} queries from {os.path.basename(file_path)}")
                if filtered_count > 0:
                    print(f"  Filtered out {filtered_count} queries with y=-1")
                total_files_processed += 1
            else:
                print(f"  Warning: {file_path} does not contain a list of queries, skipping")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    if not combined_queries:
        print("No queries found to combine!")
        return
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_file = f"combined_queries_{timestamp}.json"
    
    # If output_file is not an absolute path, save in current directory
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.getcwd(), output_file)
    
    # Save combined queries
    try:
        with open(output_file, 'w') as f:
            json.dump(combined_queries, f, indent=2)
        
        print(f"\nSuccessfully combined {len(combined_queries)} queries from {total_files_processed} files")
        if total_queries_filtered > 0:
            print(f"Filtered out {total_queries_filtered} queries with invalid y=-1 values")
        print(f"Output saved to: {output_file}")
        
        # Show some statistics
        if combined_queries:
            sizes = [len(q.get('triples', [])) for q in combined_queries]
            object_counts = [q.get('n_objects_instantiated', 0) for q in combined_queries]
            y_values = [q.get('y', 0) for q in combined_queries]
            
            print(f"\nStatistics:")
            print(f"  Total queries: {len(combined_queries)}")
            print(f"  Query sizes: {min(sizes)} - {max(sizes)} triples")
            print(f"  Object instantiation: {min(object_counts)} - {max(object_counts)} per query")
            print(f"  Y values range: {min(y_values)} - {max(y_values)}")
            
    except Exception as e:
        print(f"Error saving combined file: {e}")

if __name__ == "__main__":
    # Default pattern matches all star query files
    pattern = ""
    output_file = ""
    
    # Simple command line argument handling
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Combining files matching pattern: {pattern}")
    combine_query_files(pattern, output_file) 