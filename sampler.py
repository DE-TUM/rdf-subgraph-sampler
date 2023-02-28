import argparse
import sys
from samplers import star_query_generator as star_sampler
from samplers import path_query_generator as path_sampler


def get_options():

    # Defining arguments
    parser = argparse.ArgumentParser(description="Query sampler over RDF KGs")
    parser.add_argument("-e", "--endpoint",
                        help="URL of the SPARQL endpoint where the KG is hosted (required)")
    parser.add_argument("-q", "--queries",
                        help="Maximum number of queries to generate",
                        type=int,
                        default=5)
    parser.add_argument("-n", "--size",
                        help="Number of triple patterns in the queries",
                        type=int,
                        default=2)
    parser.add_argument("-s", "--shape",
                        help="Shape of subgraphs to generate",
                        choices=["path", "star"],
                        default="star")
    parser.add_argument("-d", "--dataset",
                        help="Dataset name (optional, used for output file)",
                        default="none")
    args = parser.parse_args()

    # Handling mandatory arguments.
    if not args.endpoint:
        print("error: no server specified. Use argument -s to specify the address of a server.")
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = get_options()

    if args.shape == 'star':
        star_sampler.get_queries(None, args.dataset, args.size, args.queries, args.endpoint)
    elif args.shape == 'path':
        path_sampler.get_queries(None, args.dataset, args.size, args.queries, args.endpoint)