# Density Peaks CSV Export
Calculate and write daily and monthly peak counts to CSV.

By default, queries all spaces associated with an organization. This can optionally
be filtered via space tags.

Configurable to calculate and store DAILY or MONTHLY peaks.

## Setup (Requires Python3)
```bash
pip install -r requirements.txt
```

#### Usage
```bash
density_peaks.py [-h] -s START_DATE -e END_DATE -t TOKEN [-p PEAK_TYPE]
                        [--tag TAG]

optional arguments:
  -h, --help            show this help message and exit
  -s START_DATE, --start-date START_DATE
                        Start date (local) of peaks query: YYYY-MM-DD
  -e END_DATE, --end-date END_DATE
                        End date (local) of peaks query. Format: YYYY-MM-DD
  -t TOKEN, --token TOKEN
                        Density API token (read-only preferred)
  -p PEAK_TYPE, --peak-type PEAK_TYPE
                        Peak type (DAILY or MONTHLY)
  --tag TAG             Filter Density spaces by tag name
```

#### Example
```bash
./occupancy_peak.py -s=2019-10-01 -e=2019-10-24 -p=DAILY -t=tok_123123123123123 --tag=conference_room
```
