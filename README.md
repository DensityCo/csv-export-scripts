# Density Peaks CSV Export
Calculate and write daily and monthly peak counts to CSV.

By default, queries all spaces associated with an organization. This can optionally
be filtered via space tags.

Configurable to calculate and store DAILY or MONTHLY peaks.

## Setup (Requires python 3)
`pip install -r requirements.txt`

## Running

`python density_peaks.py [ARGS]`

#### Args
- `-s --start-date`: Start date (local) of peaks query. Format: YYYY-MM-DD
- `-e --end-date`: Start date (local) of peaks query. Format: YYYY-MM-DD
- `-p --peak-type`: Peak type (row size) - DAILY (default) or MONTHLY
- `-t --token`: Density API token.
- `--tag`: Optional

#### Example
`python density_peaks.py -s=2019-01-01 -e=2019-01-08 -p=DAILY -t={TOKEN} --tag=conference_room`
