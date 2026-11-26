"""BGPgen: samples SPARQL basic graph pattern queries with exact cardinalities.

    import bgpgen

    graph = bgpgen.Graph("graph.nt")
    queries = graph.sample("?x B ?y, ?y B ?z, ?z B ?x", num=1000)
    paths = graph.sample_paths(3, num=1000, seed=7)
    stars = graph.sample_stars(4, num=1000)

Every call returns a list of dicts with the keys "query", "triples", "x" and
"y" (the number of results). These are the records that the bgpgen binary
writes as JSON, and the options are its command line options with "_" in
place of "-" (`bgpgen.DEFAULTS` lists them with their defaults).
"""

from importlib.metadata import PackageNotFoundError, version

from ._bgpgen import DEFAULTS, BgpgenError, Graph

try:
    __version__ = version("bgpgen")
except PackageNotFoundError:  # imported from a build folder
    __version__ = "unknown"

__all__ = ["Graph", "BgpgenError", "DEFAULTS", "__version__"]
