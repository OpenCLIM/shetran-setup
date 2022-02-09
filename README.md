# SHETRAN GB Setup


[![release](https://img.shields.io/github/v/release/openclim/shetran-setup)](https://github.com/OpenCLIM/shetran-setup/releases/latest)
[![build](https://github.com/OpenCLIM/shetran-setup/actions/workflows/build.yml/badge.svg)](https://github.com/OpenCLIM/shetran-setup/actions/workflows/build.yml)

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