import sys
from samplers import star_query_generator as star_sampler
# Import the file-based version
from samplers import star_query_generator_in_memory as file_sampler_in_memory
from samplers import path_query_generator as path_sampler
from samplers import path_query_generator_in_memory as path_sampler_in_memory
#from samplers import complex_query_generator as complex_sampler

# Import configuration
from sampler_config import *


if __name__ == "__main__":  
    if not ENDPOINT and not IN_MEMORY:
        print("error: no ENDPOINT specified. Please set the ENDPOINT variable to specify the address of a server.")
        sys.exit(1)

    if SHAPE == 'star':
        if IN_MEMORY:
            print("Using In Memory star generation...")
            file_sampler_in_memory.get_queries(None, DATASET, SIZE, QUERIES, ENDPOINT, 
                                         use_cache=USE_CACHE, min_objects_instantiated=MIN_OBJECTS_INSTANTIATED,
                                         max_objects_instantiated=MAX_OBJECTS_INSTANTIATED,
                                         rdf_file=RDF_FILE_PATH, p_predicate=P_PREDICATE, get_cardinality=GET_CARDINALITY)
        else:
            print("Using endpoint-based star generation...")
            star_sampler.get_queries(None, DATASET, SIZE, QUERIES, ENDPOINT, use_cache=USE_CACHE)
    elif SHAPE == 'path':
        if IN_MEMORY:
            print("Using In Memory path generation...")
            path_sampler_in_memory.get_queries(
                rdf_file=RDF_FILE_PATH,
                dataset_name=DATASET,
                n_triples=SIZE,
                n_queries=QUERIES,
                endpoint_url=ENDPOINT if GET_CARDINALITY else None,
                outfile=True,
                get_cardinality=GET_CARDINALITY,
                use_cache=USE_CACHE,
                sampling_method=SAMPLING_METHOD,
                enable_timing=False,
                p_edge=P_EDGE,
                p_node=P_NODE,
                p_start_end=P_START_END
            )
        else:
            print("Using endpoint-based path generation...")
            path_sampler.get_queries(None, DATASET, SIZE, QUERIES, ENDPOINT)
    #elif SHAPE == 'flower' or SHAPE == 'snowflake':
    #    complex_sampler.get_queries(None, DATASET, SHAPE, SIZE, QUERIES, ENDPOINT)