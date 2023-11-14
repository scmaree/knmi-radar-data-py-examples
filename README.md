# knmi-radar-data-py-examples

_Disclaimer: I am not affiliated with the KNMI. It just took me quite some time to get the relevant information out of the KNMI radar data,
so I want to share a walkthrough._

This repo contains python examples on how to extract buienradar-like data from the
Royal Netherlands Meteorological Institute (KNMI) API radar data.

# Getting started
To get started using conda, run the script to create a python environment with the required dependencies:
```shell
./init_python_env.sh
```
If you're on windows, use `init_python_env_windows.bat` instead.

# Request your KNMI API key
Get your API key at https://developer.dataplatform.knmi.nl/get-started#obtain-an-api-key

Save your key to a file called `.env` in the current dir or any parent dir,
containing a line with contents `knmi_api_key=XXXXXXXXXX`, following the example in the `.example` file.


# Check the notebooks
- [KNMI Radar data](notebooks/knmi_radar_data.ipynb)
