# subgraph-sampler 
Sampling approach to obtain subgraph templates (or conjunctive queries) from RDF knowledge graphs and their cardinality. 

## Preparation 
* Download or clone this repository. 
* Load the RDF knowledge graph in a SPARQL endpoint. 
* Install the Python libraries: `tqdm` and `requests`   

## Usage 
```
sampler.py -e endpoint -s shape 
```
The sampler supports several options: 
* `-q` Maximum number of queries to generate
* `-n` Number of triple patterns in the queries
* `-s` Shape of subgraphs to generate. Options `star` or `path`
* `-d` Dataset name (optional, used for output file)

## Further configurations 
The samplers have several parameters to guide the sample strategies. 
Further description about these parameters and their values will be provided here soon. 

