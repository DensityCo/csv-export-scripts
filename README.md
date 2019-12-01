# Density Peaks CSV Export
Calculate and write daily and monthly peak counts to CSV.

By default, queries all spaces associated with an organization. This can optionally
be filtered via space tags.

Configurable to calculate and store DAILY or MONTHLY peaks.

## Setup (Requires Python3)
```bash
pip install -r requirements.txt
```


## Occupancy Peaks
#### Usage
```bash
density_peaks.py [-h] -s START_DATE -e END_DATE -t TOKEN [-i INTERVAL]
                        [--tag TAG]

optional arguments:
  -h, --help            show this help message and exit
  -s START_DATE, --start-date START_DATE
                        Start date (local) of peaks query: YYYY-MM-DD
  -e END_DATE, --end-date END_DATE
                        End date (local) of peaks query. Format: YYYY-MM-DD
  -t TOKEN, --token TOKEN
                        Density API token (read-only preferred)
  -p INTERVAL, --peak-type INTERVAL
                        Peak type (DAILY or MONTHLY)
  --tag TAG             Filter Density spaces by tag name
```

#### Example
```bash
./occupancy_peak.py -s=2019-10-01 -e=2019-10-24 -i=DAILY -t=tok_123123123123123 --tag=conference_room
```

## Schedule Peaks
Calculates and writes occupancy peaks for a supplied weekly schedule, from a given start to end date.

#### Usage
```bash
schedule_peak.py [-h] -s START_DATE -e END_DATE -t TOKEN -sch SCHEDULE -sid SPACE_ID

optional arguments:
  -h, --help            show this help message and exit
  -s START_DATE, --start-date START_DATE _Start date (local) of peaks query: YYYY-MM-DD_
  -e END_DATE, --end-date END_DATE _End date (local) of peaks query. Format: YYYY-MM-DD_
  -t TOKEN, --token TOKEN _Density API token (read-only preferred)_
  -sch SCHEDULE, --schedule SCHEDULE _Path to the schedule csv file_
  -sid SPACE_ID, --space-id SPACE_ID _Space ID to pull data from_
```

The schedule (`-sch --schedule`) should be a path to a CSV file with weekly schedule information. It should contain this structure:

```csv
Class,Day of week,Weekday number,Start,End
```

Where:
  - `Class` = Class name
  - `Day of week` = Day of week name (`Sunday`)
  - `Weekday number` = 0-6 (Monday-Sunday)
  - `Start` and `End` = parsable time objects (`6:00 PM` or `18:00`).

#### Example
```bash
./schedule_peak.py -s=2019-10-01 -e=2019-10-24 -t=tok_123123123123123 -sid=spc_12312312313 -s=/path/to/schedule.csv
```
