# Information Retrieval final project

This repository contains the implementation of a recommender system. Performing collaborative filtering using matrix factorisation, the system provides movies recommendations using the MovieLens [ml-latest-small](https://grouplens.org/datasets/movielens/) dataset.


## Project organisation

The source code is located in the `source/` folder, while the dataset in the `ml-latest-small/` folder.

The code is organised as follows:
* `main.ipynb` presents the functionalities of the recommender system. It is presented in a notebook format, to provide a more convenient interaction.
* `data_preparation.py` contains the code to load and parse the dataset, building the matrices the system requires;
* `factorisation.py` implements the Weighted Alternating Least Squares algorithm for matrix factorisation;
* `recommender.py` contains the `recommenderSystem` class, which implements the functionalities of the system;
* `utilities.py` contains utility functions to measure the execution of functions and to save/load a recommender system to/from file.

Additionally, the `source/` code contains a pre-trained system, stored in `rec.npz`.

## Notes

`main.ipynb` contains indications for expected execution time on demanding cells. This expectations are referred to and Intel i7-8550U CPU and might be unreliable, depending on the available computing power.
