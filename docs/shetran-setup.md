## SHETRAN GB Setup

Sets up input data using UKCP18 12km projections for rainfall, PET and temperature. Either 1980-2010 (control) or 2040-2070 (future) scenarios can be modelled using 1 of 12 ensemble members. Any catchment mask within the UK can be used to generate SHETRAN inputs.


## Parameters

| name            | title           | description                                                 |
|:----------------|:----------------|:------------------------------------------------------------|
| CATCHMENT_ID    | Catchment ID    | The ID number of the catchment, used for naming input files |
| TIME_HORIZON    | Time horizon    | Whether to model the control or future time horizon         |
| ENSEMBLE_MEMBER | Ensemble member | The ID of the UKCP18 ensemble member to use                 |

## Dataslots

| path        | name       | description                                                                                                                   |
|:------------|:-----------|:------------------------------------------------------------------------------------------------------------------------------|
| inputs/mask | Mask       | Mask of the catchment to set up. This must be located in Great Britain.                                                       |
| inputs      | Input data | A single TAR file containing PET, precipitation and temperature from UKCP18 and static data from CAMELS-GB and other sources. |

## Outputs

| name    | description                                                             |
|:--------|:------------------------------------------------------------------------|
| outputs | SHETRAN input files. All file names are prefixed with the CATCHMENT_ID. |