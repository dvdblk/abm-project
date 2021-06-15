# Minority Game (ABM Project - FS21)

## Getting Started

### Directory Structure

```
.
├── README.md
├── environment.yml
├── lib                                       # lib package, contains MG logic
│   ├── agents                                # agents package, Agent related code
│   └── minority_game.py, strategies.py, ...  # modules
└── notebooks                                 # notebooks folder
    ├── allow_local_imports.py
    └── example.ipynb                         # example notebook
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

Open conda shell: 
on Mac: `source ~/<path to anaconda3>/bin/activate`
on Windows: search and open **anaconda3 prompt**

Activate the environment:

`conda activate abm-project`

Open `notebooks/example.ipynb` and check the example of how to run stuff in this repo.

Deactivate Environment:

`conda deactivate`
