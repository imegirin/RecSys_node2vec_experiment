Backbone of this project has been derived from https://github.com/aditya-grover/node2vec.
Code has been modified to allow execution in python-3.x.
Movielens dataset has been downloaded from https://grouplens.org/datasets/movielens/ MovieLens 1M Dataset.
MappingMovielens2DBPedia-1.2.tsv has been borrowed from https://github.com/sisinflab/LODrecsys-datasets/tree/master/Movielens1M.

To execute code run following command while in main directory:

    python src/generate_graph.py

Preformance of system will be measured based on previously generated data (graph, vectors, even testset).
To run whole pipeline use command:

    python src/generate_graph.py --regenerate

You can also get help with:

    python src/generate_graph.py --help
