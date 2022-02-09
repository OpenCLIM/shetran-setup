# SHETRAN GB Setup

This DAFNI model generates input files for SHETRAN based on a catchment mask.

## Usage 
```
docker build -t shetran-setup .
docker run -v "<absolute_path_of_data_directory>:/data" -e CATCHMENT_ID=1001 -e TIME_HORIZON=future -e ENSEMBLE_MEMBER=01 --name shetran-setup shetran-setup
```
or
```
python setup_funcs.py
```
## Documentation
[shetran-setup.md](docs/shetran-setup.md)

To build the documentation:
```
cd docs
python build_docs.py
```