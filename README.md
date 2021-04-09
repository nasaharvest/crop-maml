# Land-type mapping

A pixel-wise land type classifier, which (currently) classifies a pixel as cropland or not.

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

#### Earth Engine (recommended)

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
mypy src  # type checking
```
