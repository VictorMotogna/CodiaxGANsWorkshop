
# Generating Faces with GANs in Tensorflow

## Slack Channel

> First and foremost, join us at __[GANs Workshop Slack Channel](https://join.slack.com/t/codiax-gansworkshop/shared_invite/enQtMjcwOTY4NDczMDQxLTFjZjUyNzVkZWExYjJiZjJjYWI1ZjQ2OGFlYmVlMWY0MjRmMjI3YzYzNDIyMmEyYTc3NzEwZmMxNGM1NjgxZWM)__ so that we can keep in sync. Most probably, this channel will be the starting point for a local community of developers interested in **hands-on AI & ML** stuff.

## Data Set Download instructions

For this workshop we will use the __[MNIST](http://yann.lecun.com/exdb/mnist/)__ and __[Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)__ datasets. You can download them from __[this](https://drive.google.com/open?id=1ERLFgfDqVEJwG4R5mHihru5DRIx3cs8I)__ google drive folder. The folder contains the two `img_align_celeba` and `mnist` ziped folders. Please create a `data` folder in the project repository's root directory, copy the two dataset folders there and extract them.

The final paths relative to the project root should be: `./data/img_align_celeba` and `./data/mnist`.

## Installation Instructions

### Anaconda or Miniconda

We will be using the __[Anaconda](https://www.anaconda.com/what-is-anaconda/)__ platform for this workshop. Anaconda is
a *distribution of packages* built for *data science* that comes with **conda**, a *CLI packages* and *environments manager*.

1. Install **Conda CLI**: we are mainly interested in **conda**, therefore you have two options:
* Install __[Miniconda](https://conda.io/docs/glossary.html#miniconda-glossary)__ - a mini version of Anaconda that includes only **conda and its dependencies**
  * Download the Python 3.6 installer for your OS from __[here](https://conda.io/miniconda.html)__ and follow __[these](https://conda.io/docs/user-guide/install/index.html)__ instructions to install it.
* Install the entire __[Anaconda](https://conda.io/docs/glossary.html#anaconda-glossary)__ ecosystem which contains **conda** plus over *720 open source packages*
  * Download the Python 3.6 installer for your OS and follow the instructions from __[here](https://www.anaconda.com/download/)__.  
2. Update all the packages in the default *root* environment: `conda upgrade --all`
  ###### If you get "conda command not found" add `export PATH="/Users/username/anaconda/bin:$PATH"` to your bash config file.

#### Managing Packages

Most used commands:
* `conda install package_name`
* `conda update package_name`
* `conda remove package_name`
* `conda search search_term`
* `conda list` - list all installed packages

#### Managing Environments
Most used commands:
* `conda create -n env_name list_of_packages`
* `conda env export > environment.yaml` - save packages to .yaml file
* `conda env create -f environment.yaml` - create env from __[yaml](http://www.yaml.org/)__ file
* `source activate my_env` (Mac/Linux)  OR `activate my_env` (Windows) - activate environment
* `source deactivate` (Mac/Linux) OR `deactivate` (Windows) - deactivate environment
* `conda env list` - list all environments
* `conda env remove -n env_name`

For more info check __[this](https://conda.io/docs/user-guide/tasks/index.html)__ and __[this](https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/)__

## Environment Creation & Dependencies

For this workshop create a new environment using the provided gans_workshop.yaml by running:
* `conda env create -f .environments/sgans_environment_OS.yml`

Activate the new created environment using:
* `source activate gans_workshop` (Mac/Linux)
* `activate gans_workshop` (Windows)

## Start Test Notebook
Then run the following command to start a test __[jupyter notebook](http://nbviewer.jupyter.org/)__:
* `jupyter notebook GANsWorkshop.ipynb`
