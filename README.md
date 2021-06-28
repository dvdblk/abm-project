# Minority Game (ABM Project - FS21)
This repository contains the source code and results for the paper "The Minority Game Simulation & Implications".


## Getting Started

### Directory Structure

Here is a high-level overview of the repositories and some files in this repo:
```
.
├── LICENSE
├── README.md
├── environment.yml
├── lib/                                # lib package, contains source code
│   ├── agents/                         # agents package, Agent related code
│   ├── minority_game.py                # MG
│   ├── minority_game_vectorized.py     # Vectorized version of MG
│   └── plots.py, strategies.py, ...    # modules
├── examples/                           # contains notebooks that show some functionality
│   ├── distances.ipynb
│   └── example.ipynb
└── scenarios/                          # contains notebooks that were used in the paper
    ├── Scenario_1.ipynb
    ├── Scenario_2.ipynb
    └── out/                            # contains all the plots produced or referred by the notebooks
```

### Setting up the environment (optional)
Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or update conda to the latest version:

`conda update conda`

Run this from the root folder:

`conda env create --file environment.yml`

Activate the environment:

`conda activate abm-project`

#### Updating the dependencies
If you need to **update** one of the conda packages:

`conda env update --file environment.yml`

### Running the project

If you want to run the project you can either re-run the scenarios in the `scenarios/` folder or take a look at some smaller examples in the `examples/` folder.

Both folders contain jupyter notebooks with most of the code imported from `lib/`.
