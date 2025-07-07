#!/usr/bin/env python3
"""
File-based star query generator that uses direct RDF file parsing.
This replaces the SPARQL endpoint approach with file parsing for better performance.
"""

import sys
import gzip
import re
import json
import random
import requests
import hashlib
import time
from collections import Counter, defaultdict
from datetime import datetime
import os
from tqdm import tqdm

# Configuration
CARDINALITY_TIMEOUT = 10  # Timeout for cardinality queries in seconds

def parse_nt_line(line):
    """Parse a single N-Triples line and extract subject, predicate, object."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    

    pattern = r'(<[^>]+>)\s+(<[^>]+>)\s+(.+?)\s*\.$'
    match = re.match(pattern, line)
    
    if match:
        subject = match.group(1).strip()
        predicate = match.group(2).strip()
        object_part = match.group(3).strip()
        
        if subject.startswith('<') and subject.endswith('>'):
            subject = subject[1:-1]
        if predicate.startswith('<') and predicate.endswith('>'):
            predicate = predicate[1:-1]
        
        return subject, predicate, object_part
    
    return None

def load_or_parse_rdf_file(file_path, use_cache):
    """Load cached data or parse RDF file from scratch."""
    cache_file = file_path + ".cache.json"
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data['subject_triples'], data['subject_counts'], data['total_triples']
        except Exception as e:
            print(f"Cache loading failed: {e}, parsing file...")
    
    print(f"Parsing RDF file: {file_path}")
    
    subject_triples = defaultdict(list)  # Store all (predicate, object) pairs for each subject
    subject_counts = Counter()  # Count total triples per subject
    total_triples = 0
    
    # Handle compressed files
    if file_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    try:
        with open_func(file_path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"Processed {line_num:,} lines, found {total_triples:,} triples...")
                
                parsed = parse_nt_line(line)
                
                if parsed:
                    subject, predicate, obj = parsed
                    subject_triples[subject].append((predicate, obj))  # Store predicate-object pairs
                    subject_counts[subject] += 1  # Count total triples, not unique predicates
                    total_triples += 1
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None, 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, 0
    
    print(f"Finished parsing. Total triples: {total_triples:,}")
    print(f"Unique subjects: {len(subject_triples):,}")
    
    # Cache the results
    if use_cache:
        print(f"Caching data to {cache_file}...")
        cache_data = {
            'subject_triples': dict(subject_triples),
            'subject_counts': dict(subject_counts),
            'total_triples': total_triples,
            'timestamp': datetime.now().isoformat()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    return subject_triples, subject_counts, total_triples

def find_suitable_stars(subject_triples, subject_counts, min_size):
    """Find subjects that have at least min_size triples."""
    suitable_stars = []
    
    for subject, total_triple_count in subject_counts.items():
        if total_triple_count >= min_size:  # At least the requested size
            triples = subject_triples[subject]  # Keep all predicate-object pairs including duplicates
            
            # Count unique predicates
            unique_predicates = list(set(pred for pred, obj in triples))
            
            # Only include if we have enough total triples
            if len(triples) >= min_size:
                suitable_stars.append({
                    'subject': subject,
                    'triples': triples,  # All predicate-object pairs with duplicates
                    'star_size': total_triple_count,
                    'unique_predicates': len(unique_predicates)
                })
    
    print(f"Found {len(suitable_stars)} subjects with stars of size >= {min_size}")
    return suitable_stars

def get_query_cardinality(query, endpoint_url, timeout=CARDINALITY_TIMEOUT):
    """
    Get the cardinality of a query by running COUNT(*) query.
    
    Args:
        query (str): The original SELECT query
        endpoint_url (str): SPARQL endpoint URL
        timeout (int): Query timeout in seconds
    
    Returns:
        int: Query cardinality, or -1 if failed
    """
    
    if not endpoint_url:
        print("WARNING: No endpoint URL provided - cannot compute cardinality")
        return -1
    
    # Extract WHERE clause from the original query
    where_start = query.upper().find('WHERE')
    if where_start == -1:
        print("WARNING: No WHERE clause found in query - cannot compute cardinality")
        return -1
    
    where_clause = query[where_start:]
    count_query = f"SELECT (COUNT(*) as ?count) {where_clause}"
    

    
    try:
        response = requests.get(
            endpoint_url,
            params={
                'query': count_query,
                'format': 'json'
            },
            timeout=timeout
        )

        
        if response.status_code == 200:
            result = response.json()
            bindings = result["results"]["bindings"]
            if bindings and 'count' in bindings[0]:
                return int(bindings[0]['count']['value'])
        
        print("WARNING: No result from endpoint - returning -1")
        return -1
        
    except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError, KeyError):
        print("WARNING: Error computing cardinality - returning -1")
        return -1

def hash_query_pattern(predicates, object_instantiation_pattern):
    """Create a hash for a query pattern including predicate set and object instantiation pattern."""
    # Sort predicates to ensure consistent hashing
    sorted_predicates = sorted(predicates)
    # Create a string representation including object instantiation pattern
    pattern_str = '|'.join(sorted_predicates) + f"_OBJ:{sorted(object_instantiation_pattern)}"
    return hashlib.md5(pattern_str.encode('utf-8')).hexdigest()

def generate_star_query(star_data, n_triples, prob_predicate=1.0, min_objects_instantiated=0, 
                       max_objects_instantiated=0, endpoint_url=None, get_cardinality=False, max_object_combinations=5):
    """
    Generate a star query from a star data structure with configurable object instantiation.
    
    Args:
        star_data: Dictionary with 'subject' and 'triples' (predicate-object pairs)
        n_triples: Number of triple patterns to generate
        prob_predicate: Probability of instantiating predicates
        min_objects_instantiated: Minimum number of objects to keep instantiated
        max_objects_instantiated: Maximum number of objects to keep instantiated
        endpoint_url: SPARQL endpoint URL for cardinality calculation
        get_cardinality: Whether to calculate query cardinality
        max_object_combinations: Maximum number of object combinations to try
    
    Returns:
        Dictionary with query and metadata, or None if generation failed
    """
    triples = star_data['triples']
    
    # Take exactly n_triples from the available ones (with duplicates)
    if len(triples) < n_triples:
        return None
    
    # Take the first n_triples (preserving duplicates as they appear)
    selected_triples = triples[:n_triples]
    selected_predicates = [pred for pred, obj in selected_triples]
    
    # Validate and set object instantiation range
    min_objects_instantiated = max(0, min(min_objects_instantiated, n_triples))
    max_objects_instantiated = max(0, min(max_objects_instantiated, n_triples))
    
    # Ensure min <= max
    if min_objects_instantiated > max_objects_instantiated:
        min_objects_instantiated, max_objects_instantiated = max_objects_instantiated, min_objects_instantiated
    
    # Randomly choose the number of objects to instantiate for this query
    n_objects_instantiated = random.randint(min_objects_instantiated, max_objects_instantiated)
    
    # Try different object instantiation combinations
    for attempt in range(max_object_combinations):
        # Randomly choose which objects to instantiate
        if n_objects_instantiated > 0:
            instantiated_indices = random.sample(range(n_triples), n_objects_instantiated)
        else:
            instantiated_indices = []
        
        # Generate the query template
        triple_patterns = []
        entities = []  # Track instantiated entities (predicates)
        instantiated_objects = []  # Track instantiated objects
        
        for i, (predicate, obj) in enumerate(selected_triples):
            # Handle predicate instantiation
            if random.random() <= prob_predicate:
                # Instantiate the predicate
                pred_part = f"<{predicate}>"
                entities.append(predicate)
            else:
                # Keep predicate as variable
                pred_part = f"?p{i}"
            
            # Handle object instantiation
            if i in instantiated_indices:
                obj_part = obj
                instantiated_objects.append(obj)
            else:
                # Use variable for object
                obj_part = f"?o{i}"
            
            triple_patterns.append(f"?s {pred_part} {obj_part}")
        
        # Join triple patterns
        where_clause = " . ".join(triple_patterns)
        final_query = f"SELECT * WHERE {{ {where_clause} }}"
        
        cardinality = -1
        if get_cardinality and endpoint_url:
            cardinality = get_query_cardinality(final_query, endpoint_url)
        
        triples_list = []
        for pattern in triple_patterns:
            pattern = pattern.strip()
            if pattern.startswith('?s '):
                rest = pattern[3:]  # Remove '?s '
                
                # Find the predicate part
                if rest.startswith('<'):
                    # Predicate is a URI
                    pred_end = rest.find('> ')
                    if pred_end != -1:
                        pred_part = rest[:pred_end + 1]
                        obj_part = rest[pred_end + 2:]
                        triples_list.append(['?s', pred_part, obj_part])
                else:
                    # Predicate is a variable
                    parts = rest.split(' ', 1)
                    if len(parts) >= 2:
                        pred_part = parts[0]
                        obj_part = parts[1]
                        triples_list.append(['?s', pred_part, obj_part])
        
        # Create query pattern hash for deduplication (includes object instantiation pattern)
        query_hash = hash_query_pattern(selected_predicates, instantiated_indices)
        
        return {
            "query": final_query,
            "triples": triples_list,
            "x": entities,  # Instantiated predicates
            "y": cardinality,  # Query cardinality
            "source_subject": star_data['subject'],
            "star_size": star_data['star_size'],
            "query_hash": query_hash,  # For deduplication (includes object pattern)
            "predicates_used": selected_predicates,  # For debugging/analysis
            "instantiated_objects": instantiated_objects,  # Instantiated objects
            "object_instantiation_pattern": instantiated_indices,  # Which positions have instantiated objects
            "n_objects_instantiated": n_objects_instantiated  # Actual number instantiated for this query
        }
    
    # If we couldn't generate a unique query after max_object_combinations attempts
    return None

def get_queries(graphfile, dataset_name, n_triples=10, n_queries=1000, 
                endpoint_url=None, subjects=[], get_cardinality=False, 
                outfile=True, use_cache=True, rdf_file=None, min_objects_instantiated=0, max_objects_instantiated=0,
                p_predicate=1.0):
    
    if not os.path.exists(rdf_file):
        print(f"Error: RDF file {rdf_file} not found")
        return []
    
    now = datetime.now()
    min_star_size = n_triples
    
    print(f"Generating {n_queries} star queries of size {n_triples}")
    print(f"Using RDF file: {rdf_file}")
    print(f"Minimum star size: {min_star_size}")
    print(f"Objects to instantiate: {min_objects_instantiated}-{max_objects_instantiated} per query")
    print(f"Predicate instantiation probability: {p_predicate}")
    if get_cardinality:
        print(f"Will calculate cardinalities using endpoint: {endpoint_url}")
    
    # Parse the RDF file
    subject_triples, subject_counts, total_triples = load_or_parse_rdf_file(rdf_file, use_cache)
    
    if subject_triples is None:
        print(f"Warning: No subject triples found in {rdf_file}")
        return []
    
    # Find suitable stars
    suitable_stars = find_suitable_stars(subject_triples, subject_counts, min_star_size)
    
    if not suitable_stars:
        print(f"No suitable stars found with size >= {min_star_size}")
        return []
    
    # Debug: Show statistics about suitable stars
    unique_pred_counts = [star['unique_predicates'] for star in suitable_stars]
    print(f"Stars with >= {n_triples} unique predicates: {sum(1 for count in unique_pred_counts if count >= n_triples)}")
    
    # Generate queries with deduplication
    print("Generating star queries with deduplication...")
    testdata = []
    failed_generations = 0
    cardinality_failures = 0
    duplicate_skipped = 0
    seen_query_hashes = set()  # Track seen query patterns (including object instantiation)
    
    # Create a shuffled copy of suitable stars for sampling without replacement
    available_stars = suitable_stars.copy()
    random.shuffle(available_stars)
    star_index = 0
    
    

    with tqdm(total=n_queries, desc="Generating unique queries") as pbar:
        while len(testdata) < n_queries:

                            
            # Select star without replacement
            if star_index >= len(available_stars):
                # If we've exhausted all stars, reshuffle and start over
                if duplicate_skipped == 0:
                    print("\nExhausted all available stars without finding duplicates. Dataset may have limited diversity.")
                    break
                else:
                    # Reshuffle and continue - there might be different combinations
                    random.shuffle(available_stars)
                    star_index = 0
                    print(f"\nReshuffling stars (found {duplicate_skipped} duplicates so far)...")
            
            star = available_stars[star_index]
            star_index += 1
            
            # Generate query from this star with object instantiation
            query_data = generate_star_query(star, n_triples, p_predicate, min_objects_instantiated, 
                                           max_objects_instantiated, endpoint_url, get_cardinality)
            
            if query_data:
                query_hash = query_data['query_hash']
                
                # Check if we've already seen this query pattern
                if query_hash in seen_query_hashes:
                    duplicate_skipped += 1
                    continue
                
                # This is a new unique query
                seen_query_hashes.add(query_hash)
                testdata.append(query_data)
                pbar.update(1)
                
                # Track cardinality calculation failures
                if get_cardinality and query_data['y'] == -1:
                    cardinality_failures += 1
            else:
                failed_generations += 1
            
            # Save periodically
            if outfile and len(testdata) % 100 == 0 and len(testdata) > 0:
                filename = f"{dataset_name}_stars_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{n_triples}.json"
                with open(filename, "w") as fp:
                    json.dump(testdata, fp, indent=2)
    
    print(f"- Unique queries generated: {len(testdata)}")
    print(f"- Duplicates skipped: {duplicate_skipped}")
    if failed_generations > 0:
        print(f"- Failed generations: {failed_generations}")
    if get_cardinality and cardinality_failures > 0:
        print(f"- Cardinality calculation failures: {cardinality_failures}")
    
    if len(testdata) < n_queries:
        print(f"Warning: Only generated {len(testdata)} unique queries out of {n_queries} requested")
        print(f"This might indicate limited diversity in the dataset")
    
    # Final save
    if outfile:
        filename = f"{dataset_name}_stars_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{n_triples}.json"
        with open(filename, "w") as fp:
            json.dump(testdata, fp, indent=2)
        print(f"Saved {len(testdata)} unique queries to {filename}")
    
    return testdata