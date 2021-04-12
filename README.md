# Meta-learning for crop mapping

This repository contains the implementation of "Learning to predict crop type from heterogeneous sparse labels using meta-learning", which will be published at the GeoVision workshop at CVPR 2020.

## Pipeline
The main entrypoints into the pipeline are [scripts](scripts). Specifically:

* [scripts/export.py](scripts/export.py) exports data (locally, or to Google Drive, depending on what is being exported)
* [scripts/process.py](scripts/process.py) processes the raw data
* [scripts/engineer.py](scripts/engineer.py) combines the earth observation data with the labels to create (x, y) training data
* [scripts/maml.py](scripts/maml.py) trains the MAML model
* [scripts/test.py](scripts/test.py) tests the trained MAML model by finetuning it on the test datasets
* [scripts/ensemble.py](scripts/ensemble.py) takes weights saved by [test.py](scripts/test.py) and ensembles them to create maps
* [scripts/pretrain.py](scripts/pretrain.py) trains a model on all data, for a transfer learning baseline

Two crop type maps created using few positive labelled points are available on [Google Earth Engine](https://code.earthengine.google.com/39a0fedfc7ac7f21c3dcb06eab29917d):
* [Coffee map for 2019-2020 season in Luís Eduardo Magalhães municipality, Brazil](https://code.earthengine.google.com/6d348205d0313a0fdf1ebeaf14edd359)
* [Common bean map for 2019-2020 season in Busia, Kenya](https://code.earthengine.google.com/7ebf03937d5c376dd657dba1d881e789)

## Setup

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `landcover-mapping` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate landcover-mapping
```

#### Earth Engine

Earth engine is used instead of sentinel hub, because it is free. To use it, once the conda environment has been activated, run

```bash
earthengine authenticate
```

and follow the instructions. To test that everything has worked, run

```bash
python -c "import ee; ee.Initialize()"
```

Note that Earth Engine exports files to Google Drive by default (to the same google account used sign up to Earth Engine).

Running exports can be viewed (and individually cancelled) in the `Tabs` bar on the [Earth Engine Code Editor](https://code.earthengine.google.com/).
For additional support the [Google Earth Engine forum](https://groups.google.com/forum/#!forum/google-earth-engine-developers) is super
helpful.

#### Tests

The following tests can be run against the pipeline:

```bash
pytest  # unit tests, written in the test folder
black .  # code formatting
```
