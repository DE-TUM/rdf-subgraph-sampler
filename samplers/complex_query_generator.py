import json
import math

from tqdm import tqdm
import requests
import random
from datetime import datetime
import star_query_generator
import path_query_generator

# ENDPOINT_LIMIT: Should be high for short path queries, adjust according to the capacity of the endpoint
ENDPOINT_LIMIT = 500
# QUERIES_PER_SEED: Set up to 1 for short queries, 3 for long queries (with length >= 5)
STARS_PER_SEED = 5
# FINAL_QUERY_TIMEOUT: Can be set up higher for long queries
FINAL_QUERY_TIMEOUT = 5
# Configurations for path and star generators
path_query_generator.SEED_BATCHES = 50
path_query_generator.P_NODE = 0
path_query_generator.P_START_END = 0
path_query_generator.P_START_START = 0
star_query_generator.P_PREDICATE = 1.0
star_query_generator.P_OBJECT = 0.5
star_query_generator.SEED_SUBJECTS = 20000


def create_stars(n_triples, endpoint_url, subjects, s_var, o_var):
    stars = star_query_generator.get_testdata(None, "", n_triples,
                                              STARS_PER_SEED, endpoint_url, subjects, get_cardinality=False, outfile=False)
    queries = []
    for s in stars:
        query = ""
        triples_sample = random.sample(s['triples'], random.randint(1, n_triples))
        for t in triples_sample:
            if t[2].startswith('?o'):
                query += '?' + s_var + ' ' + t[1] + ' ' + t[2].replace('o', o_var) + ' . '
            else:
                query += '?' + s_var + ' ' + t[1] + ' ' + t[2] + ' . '
        queries.append(query)

    print("Leaving create stars")
    return queries


def get_queries(graphfile, dataset_name, shape='flower', path_size=2, n_triples=1, n_queries=30000, endpoint_url=None, outfile=True, get_cardinality=True):
    now = datetime.now()

    # Get path queries
    path_queries = path_query_generator.get_testdata(None, dataset_name, path_size, n_queries=math.ceil(n_queries/2),
                                                     endpoint_url=endpoint_url, outfile=False)

    testdata = []
    for i in tqdm(range(0, n_queries)):
        j = random.randint(0, len(path_queries)-1)
        path_query = " ".join(list(map(lambda n: " ".join(n) + " .", path_queries[j]['triples']))) #path_queries[j]['query']
        rj = requests.get(endpoint_url,
                          params={'query': 'SELECT DISTINCT ?o0 ?o' + str(path_size) + ' WHERE { ' + path_query + ' }' +
                                           # " ORDER BY ASC(bif:rnd(2000000000)) " +
                                           " LIMIT " + str(ENDPOINT_LIMIT),
                                  'format': 'json'},
                          timeout=FINAL_QUERY_TIMEOUT)

        if rj.status_code == 200:
            qres = rj.json()
            qres = qres["results"]["bindings"]

            s_start = []
            s_end = []
            for d in qres:
                if d['o0']['type'] == 'uri':
                    s_start.append('<' + d['o0']['value'] + '>')
                if d['o' + str(path_size)]['type'] == 'uri':
                    s_end.append('<' + d['o' + str(path_size)]['value'] + '>')

            # Build the extreme of the snowflake / flower
            s_start = list(set(s_start))
            s_end = list(set(s_end))
            stars1 = []
            stars2 = []
            if shape == 'flower':
                if s_start:
                    stars1 = create_stars(n_triples, endpoint_url, s_start, 'o0', 'x')
                if s_end:
                    stars1 += create_stars(n_triples, endpoint_url, s_end, 'o' + str(path_size), 'y')
                if not stars1:
                    continue
            else:
                if not s_start or not s_end:
                    continue
                if s_start:
                    stars1 = create_stars(n_triples, endpoint_url, s_start, 'o0', 'x')
                if s_end:
                    stars1 += create_stars(n_triples, endpoint_url, s_end, 'o' + str(path_size), 'y')
                print("Building right stars", len(stars1), stars1)
                stars2 = create_stars(n_triples, endpoint_url, s_end, 'o' + str(path_size), 'y')
                if not stars2:
                    print("Could not build right star :(")
                    continue

            # Build final query
            print("before final query", len(stars1), len(stars2))
            for k in range(0, min(STARS_PER_SEED, len(stars1))):
                if not len(stars1):
                    break
                final_query = path_query
                if shape == 'flower':
                    final_query += stars1[k]
                else:
                    try:
                        final_query += stars2[random.randint(0,len(stars2)-1)]
                        r = requests.get(endpoint_url,
                                         params={'query': 'SELECT DISTINCT ?o0 WHERE { ' + final_query + '} ' +
                                                          'ORDER BY ASC(bif:rnd(2000000000)) ' +
                                                          'LIMIT ' + str(ENDPOINT_LIMIT),
                                                 'format': 'json'},
                                         timeout=FINAL_QUERY_TIMEOUT)
                        if r.status_code == 200:
                            rres = r.json()
                            rres = rres["results"]["bindings"]
                            s_seed = ['<' + r['o0']['value'] + '>' for r in rres if r['o0']['type'] == 'uri']
                            if not s_seed:
                                continue
                            stars1 = create_stars(n_triples, endpoint_url, s_seed, 'o0', 'x')
                            final_query += random.sample(stars1, 1)[0]
                    except requests.exceptions.ReadTimeout:
                        pass

                if not get_cardinality:
                    datapoint = {"query": 'SELECT * WHERE { ' + final_query + ' }',
                                 "triples": [elem.strip().split() for elem in final_query.split(" .")[:-1]]}
                    testdata.append(datapoint)
                    continue

                # Get cardinality of final query
                try:
                    rn = requests.get(endpoint_url,
                                      params={'query': 'SELECT COUNT(*) as ?res WHERE { ' + final_query + '}',
                                              'format': 'json'},
                                      timeout=FINAL_QUERY_TIMEOUT)
                    if rn.status_code == 200:
                        qres2 = rn.json()
                        qres2 = qres2["results"]["bindings"]
                        datapoint = {"y": int(qres2[0]["res"]["value"]),
                                     "query": 'SELECT * WHERE { ' + final_query + ' }',
                                     "triples": [elem.strip().split() for elem in final_query.split(" .")[:-1]]}
                        if datapoint['y']:
                            testdata.append(datapoint)
                            print(datapoint['y'], datapoint['query'])
                            #print(datapoint['triples'])
                        else:
                            print("EMPTY.------------------------")
                except requests.exceptions.ReadTimeout:
                    pass

        if outfile and i % 10 == 0:
            with open(dataset_name + "_" + shape + "_" + str(path_size) + "_" + str(n_triples) + "_" + now.strftime('%Y-%m-%d_%H-%M-%S_') +
                      str(n_triples) + ".json", "w") as fp:
                json.dump(testdata, fp)

    if outfile:
        with open(dataset_name + "_" + shape + "_" + str(path_size) + "_" + str(n_triples) + "_" + now.strftime('%Y-%m-%d_%H-%M-%S_') +
                  str(n_triples) + ".json", "w") as fp:
            json.dump(testdata, fp)

    print("Done:", len(testdata))


if __name__ == "__main__":
    get_queries(None, "gcare-yago", shape='snowflake', path_size=1, n_triples=2, n_queries=6000,
                 endpoint_url="http://localhost:8896/sparql", outfile=True)

