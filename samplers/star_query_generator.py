import json
from tqdm import tqdm
import requests
import random
from collections import Counter
import math
import itertools
from datetime import datetime
import os
import hashlib

# SEED_SUBJECTS: Number of subjects to sample to obtain stars
#SEED_SUBJECTS = 5000000
SEED_SUBJECTS = 5000000
# SUBJECTS_BATCH: Number of subjects to group in a single request to get stars
SUBJECTS_BATCH = 50
# ENDPOINT_LIMIT: Should be high for short path queries, adjust according to the capacity of the endpoint
ENDPOINT_LIMIT = 5000
# QUERIES_PER_SEED: Set up to 1 for short queries, 3 for long queries (with length >= 5)
QUERIES_PER_SEED = 1
#QUERIES_PER_SEED = 3
# P_INSTANTIATE: Probability of instantiating a star with objects. Set to 0.75 for long queries
#P_INSTANTIATE = 0.65
#P_INSTANTIATE = 0.80
P_INSTANTIATE = 0.
# MAX_TP_INSTANTIATE: Maximum number of triple patterns to instantiate the objects
MAX_TP_INSTANTIATE = 4
# P_OBJECT: Probability of instantiating a specific object in the star. Set to 0.75 for long queries
#P_OBJECT = 0.75
P_OBJECT = 0.
# P_OBJECT: Probability of instantiating a specific predicate. Set to 1.0 for long queries, or endpoint won't finish
P_PREDICATE = 1.
# P_PREDICATE = 0.99
# FINAL_QUERY_TIMEOUT: Can be set up higher for long queries
FINAL_QUERY_TIMEOUT = 3


def get_cache_filename(endpoint_url):
    """Generate a cache filename based on the endpoint URL"""
    endpoint_hash = hashlib.md5(endpoint_url.encode()).hexdigest()[:8]
    return f"seed_subjects_cache_{endpoint_hash}.json"

def load_cached_subjects(endpoint_url):
    """Load cached seed subjects if available"""
    cache_file = get_cache_filename(endpoint_url)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                print(f"Loaded {len(data['subjects'])} cached seed subjects from {cache_file}")
                return data['subjects']
        except (json.JSONDecodeError, KeyError):
            print(f"Cache file {cache_file} is corrupted, will regenerate")
            os.remove(cache_file)
    return None

def save_cached_subjects(endpoint_url, subjects):
    """Save seed subjects to cache"""
    cache_file = get_cache_filename(endpoint_url)
    cache_data = {
        'endpoint': endpoint_url,
        'timestamp': datetime.now().isoformat(),
        'count': len(subjects),
        'subjects': subjects
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    print(f"Cached {len(subjects)} seed subjects to {cache_file}")

def get_seed_subjects(endpoint_url):
    r = requests.get(endpoint_url,
                     params={'query': "SELECT DISTINCT ?s WHERE { ?s ?p ?o . } " +
                                      "ORDER BY ASC(bif:rnd(2000000000)) " +
                                      "LIMIT " + str(ENDPOINT_LIMIT),
                             'format': 'json'})
    res = r.json()
    res = res["results"]["bindings"]
    return ['<'+d['s']['value']+'>' for d in res]


def get_seed_stars(endpoint_url, subjects, n_triples):
    stars = {}

    for i in tqdm(range(0, math.ceil(len(subjects)/SUBJECTS_BATCH))):
        values = " ".join(list(map(lambda s: "("+s+")", subjects[i:i+SUBJECTS_BATCH])))
        r = requests.get(endpoint_url,
                         params={'query': "SELECT ?s ?p WHERE { ?s ?p ?o . VALUES (?s) { " + values + " } }" +
                                          "ORDER BY ?s ?p",
                                 'format': 'json'})
        res = r.json()
        res = res["results"]["bindings"]

        cand_stars = {}
        for elem in res:
            if elem['s']['value'] in cand_stars.keys():
                cand_stars[elem['s']['value']].append(elem['p']['value'])
            else:
                cand_stars[elem['s']['value']] = [elem['p']['value']]

    for (k, v) in cand_stars.items():
        if len(v) >= n_triples:
            for p in set(itertools.combinations(v, n_triples)):
                if p in stars.keys():
                    stars[p].append(k)
                else:
                    stars[p] = [k]
                #if v[0] == v[-1]:
                #    stars[k] = v[0:n_triples]
                #else:
                #    stars[k] = v
    return stars


def generate_template(n_triples, start=1, predicates=[]):
    where = ""
    for i in range(start, n_triples-start):
        where += " ?s ?p" + str(i) + " ?o" + str(i) + " . "

    filter = ""
    if predicates:
        for i in range(start, n_triples-start-1):
            if predicates[i] == predicates[i+1]:
                filter += " FILTER (?o" + str(i) + " < " + "?o" + str(i+1) + ") "
    return where, filter


def instantiate_predicates(bindings, query, prob=1.0):
    entities = []
    for k in range(0, len(bindings)):
        if random.random() <= prob:
            query = query.replace("?p" + str(k), "<" + bindings[k] + ">")
            entities.append(bindings[k])
    return query, entities


def instantiate_objects(bindings, query):
    entities = []
    for (var, val) in bindings.items():
        if random.random() < P_OBJECT:
            if val['type'] == 'uri':
                query = query.replace("?" + var, "<" + val['value'] + ">")
            #else:
            #    query = query.replace("?" + var, '"' + val['value'] + '"')
            entities.append(val['value'])
    return query, entities


def extend_star(query, predicate_counts, start):
    j = start
    for (k, v) in predicate_counts.items():
        for i in range(0, v):
            if random.random() < P_PREDICATE:
                query += " ?s <" + k + "> ?o" + str(i+j) + " ."
            else:
                query += " ?s ?p" + str(i + j) + " ?o" + str(i + j) + " ."
        j = j + v
    return query


def get_batch_seed_subjects(endpoint_url, use_cache=True):
    """Get seed subjects with optional caching"""
    if use_cache:
        cached_subjects = load_cached_subjects(endpoint_url)
        if cached_subjects:
            return cached_subjects
    
    print("Fetching fresh seed subjects...")
    subjects = []
    for i in tqdm(range(0, math.ceil(SEED_SUBJECTS / ENDPOINT_LIMIT))):
        subjects += get_seed_subjects(endpoint_url)
    subjects = list(set(subjects))
    
    if use_cache:
        save_cached_subjects(endpoint_url, subjects)
    
    return subjects


def get_queries(graphfile, dataset_name, n_triples=1, n_queries=30000, endpoint_url=None, subjects=[], get_cardinality=True, outfile=True, use_cache=True):
    now = datetime.now()

    # Get seed subjects to explore stars
    if not subjects:
        print("Getting {} seed subjects in {} requests".format(SEED_SUBJECTS, math.ceil(SEED_SUBJECTS/ENDPOINT_LIMIT)))
        subjects = get_batch_seed_subjects(endpoint_url, use_cache)

    # Get seed stars larger than n_triples:
    # stars is a list of pairs of the form [((predicates), [seed_entities])], where predicates is an n-triples-tuple
    stars = get_seed_stars(endpoint_url, subjects, n_triples)
    stars = list(stars.items())

    # Stars could not be found for the given subjects
    if not stars:
        return []

    # Form stars of exact length n_triples and get their cardinality
    print("Generating star queries ...")
    testdata = []
    for i in tqdm(range(0, n_queries)):
        try:
            j = random.randint(0, len(stars) - 1)

            for k in range(0, QUERIES_PER_SEED):
                predicates = stars[j][0]
                aux, _ = generate_template(n_triples, 0)
                final_query, entities = instantiate_predicates(predicates, aux, P_PREDICATE)

                # Instantiate some objects in the star
                if random.random() <= P_INSTANTIATE:
                    sample_predicates = random.sample(predicates, min([n_triples, MAX_TP_INSTANTIATE]))
                    t_where, t_filter = generate_template(len(sample_predicates), 0, sample_predicates)
                    sample_query, entities = instantiate_predicates(sample_predicates, t_where)
                    random_entity = random.randint(0, len(stars[j][1])-1)
                    r = requests.get(endpoint_url,
                                     params={'query': "SELECT * WHERE { " +
                                                      sample_query.replace("?s", "<" + stars[j][1][random_entity] + ">")
                                                      + t_filter +
                                                      " } ORDER BY ASC(bif:rnd(2000000000)) LIMIT 1",
                                             'format': 'json'},
                                     timeout=FINAL_QUERY_TIMEOUT)

                    if r.status_code == 200:
                        qres = r.json()
                        qres = qres["results"]["bindings"][0]
                        sample_query, entities = instantiate_objects(qres, sample_query)
                        predicates_to_complete = dict(Counter(predicates) - Counter(sample_predicates))
                        final_query = extend_star(sample_query, predicates_to_complete, len(sample_predicates))

                if not get_cardinality:
                    testdata.append({"query": "SELECT * WHERE { " + final_query + " }",
                                     "triples": [elem.strip().split() for elem in final_query.split(" .")[:-1]]})
                    continue

                # Get cardinality of final query
                rn = requests.get(endpoint_url,
                                  params={'query': "SELECT COUNT(*) as ?res WHERE { " + final_query + " }",
                                          'format': 'json'},
                                  timeout=FINAL_QUERY_TIMEOUT)
                if rn.status_code == 200:
                    qres2 = rn.json()
                    qres2 = qres2["results"]["bindings"]
                    datapoint = {"x": entities,
                                 "y": int(qres2[0]["res"]["value"]),
                                 "query": "SELECT * WHERE { " + final_query + " }",
                                 "triples": [elem.strip().split() for elem in final_query.split(" .")[:-1]]}
                    testdata.append(datapoint)
                    #print(datapoint['y'], datapoint['query'], len(datapoint['triples']))

        except requests.exceptions.ReadTimeout:
            pass

        if outfile and i % 100 == 0:
            with open(dataset_name + "_stars_" + now.strftime('%Y-%m-%d_%H-%M-%S_') + str(n_triples) + ".json", "w") as fp:
                json.dump(testdata, fp)

    if outfile:
        with open(dataset_name + "_stars_" + now.strftime('%Y-%m-%d_%H-%M-%S_') + str(n_triples) + ".json", "w") as fp:
            json.dump(testdata, fp)

    print("Done:", len(testdata))
    return testdata


if __name__ == "__main__":
    get_queries(None, "gcare-yago", n_triples=8, n_queries=1000,
                 endpoint_url="http://localhost:8896/sparql")

