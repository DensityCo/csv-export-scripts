import argparse
import csv
from datetime import datetime, timedelta
import itertools
import sys

from dateutil import parser, tz
import pytz
import requests


API_ROOT = 'https://api.density.io/v2'
OUTPUT_DATE_FORMAT = '%Y-%m-%d'


# ====
# API methods
# ====
def auth_header(token):
    """Auth header object for Density API"""
    return {'Authorization': 'Bearer {}'.format(token)}


def get_counts(token, space_id, start_time=None, end_time=None, interval='1d', paginate_next=None, order='ASC'):
    """Convenience method to hit the space events endpoint. Will act recursively if
    data is paginated.

    Args:
        token: Density API token
        space_id: Space ID

    Kwargs:
        start_time: UTC datetime for beginning of query
        end_time: UTC datetime for end of query
        paginate_next: URL for next pagination (will override setting initial params in request)

    Returns:
        [{...}] Counts array
    """
    if paginate_next:
        request = requests.get(paginate_next, headers=auth_header(token))
    else:
        params = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'order': 'ASC',
            'interval': interval,
            'page_size': 5000,
            'order': order
        }

        request = requests.get(
            '{}/spaces/{}/counts'.format(API_ROOT, space_id),
            params=params,
            headers=auth_header(token)
        )
    request.raise_for_status()

    response = request.json()

    if response['next'] is not None:
        response['results'].extend(get_events(
            token, space_id, paginate_next=response['next']))

    return response['results']


def pull_spaces(token, tag=''):
    """Convenience method to hit the spaces endpoint.

    Args:
        token: Density API token

    Kwargs:
        tag: space tag filter

    Returns:
        [{...}] Spaces array
    """
    params = {}

    if len(tag) > 0:
        params['tag'] = tag

    request = requests.get(
        '{}/spaces/'.format(API_ROOT),
        params=params,
        headers=auth_header(token)
    )
    request.raise_for_status()
    response = request.json()

    return response['results']


# ====
# datetime helpers
# ====
def timestamp_to_local(timestamp, local_tz):
    """Converts a UTC datetime a specified local timezone"""
    utc = parser.parse(timestamp)
    utc.replace(tzinfo=tz.gettz('UTC'))

    return utc.astimezone(tz.gettz(local_tz))


def parse_interval_as_timedelta(interval):
    """Given a formatted interval string, return it as a timedelta"""
    duration_units = {'w': 'weeks', 'd': 'days', 'h': 'hours', 'm': 'minutes', 's': 'seconds'}
    suffix = interval[-1]
    amount = int(interval[0:-1])
    return timedelta(**{duration_units[suffix]: amount})


def split_time_range_into_subranges_with_same_offset(time_zone, start, end, params={}):
    """Method which divides a range of local time into "constant-UTC-offset" subranges,
    such that each subrange can be passed as UTC (with the provided interval)
    to the Density API.

    The responses to these queries can then be merged to represent e.g. "daily intervals",
    even though the days of DST transitions are not the same length as other days.

    Args:
        - time_zone (str): local time zone
        - start (datetime): local start date
        - end (datetime): local end date

    Kwargs:
        - params ({interval, order}): Density API count endpoint params

    Returns:
        - [{start: end: gap:}] array of query-ready sub ranges
    """
    tz = pytz.timezone(time_zone)

    # Convert start and end into UTC times
    start = tz.localize(start).astimezone(pytz.UTC)
    end = tz.localize(end).astimezone(pytz.UTC)
    results = []

    # Same defaults as API
    params['interval'] = params.get('interval', '1h')
    params['order'] = params.get('order', 'asc')
    interval = parse_interval_as_timedelta(params['interval'])
    reverse = params['order'].lower() == 'desc'

    # Validate start and end timestamps
    if (start >= end):
        raise ValueError("Start must be before end!")

    # Create a list of DST transitions within this range of (local) time
    transitions = []
    for i in range(len(tz._utc_transition_times)):
        time = pytz.UTC.localize(tz._utc_transition_times[i])
        if start < time and time < end:
            shift = tz._transition_info[i-1][0] - tz._transition_info[i][0]
            transitions.append({'time': time, 'shift': shift})

    # Save the last segment and interval boundaries that we've processed so far
    last_segment = end if reverse else start
    last_interval = last_segment

    # Generate necessary segments to avoid each DST boundary
    while len(transitions) > 0:
        # Depending on the order of results, we pull from either end of the transitions array
        transition = transitions.pop() if reverse else transitions.pop(0)
        transition_point = transition['time']
        transition_shift = transition['shift']

        # Skip by "interval" in the correct order until we've either passed or reached the transition
        while last_interval > transition_point if reverse else last_interval < transition_point:
            last_interval = last_interval - interval if reverse else last_interval + interval

        # Create a subrange from the last segment to this transition, preserving "time forwards" order
        results.append({
            'start': transition_point if reverse else last_segment,
            'end': last_segment if reverse else transition_point,
            'gap': False
        })

        # If querying days or weeks, shift the next interval to align with the new offset
        if params['interval'][-1] in ['d', 'w']:
            last_interval = last_interval - transition_shift if reverse else last_interval + transition_shift

        # If there is a gap before the next interval, it will need to be fetched separately
        if last_interval != transition_point:
            results.append({
                'start': last_interval if reverse else transition_point,
                'end': transition_point if reverse else last_interval,
                'gap': True
            })

        # Continue from the last even interval
        last_segment = last_interval

    # Add the last interval if necessary
    if start < last_segment if reverse else end > last_segment:
        results.append({
            'start': start if reverse else last_segment,
            'end': last_segment if reverse else end,
            'gap': False
        })

    # Return array of subranges
    return results

def pull_counts_for_time_ranges(token, space_id, time_ranges):
    """Pulls count buckets from the density API given a space and set of DST safe
    time ranges.
    """
    counts_array = []

    for subrange in time_ranges:
        counts = get_counts(
            token,
            space_id,
            start_time=subrange['start'],
            end_time=subrange['end'],
            interval='1d'
        )

        if subrange['gap'] and len(counts) > 0:
            gap_interval= counts[0]['interval']
            last_interval = counts_array[-1]['interval']

            last_interval['analytics']['entrances'] += gap_interval['analytics']['entrances']
            last_interval['analytics']['exits'] += gap_interval['analytics']['exits']
            last_interval['analytics']['events'] += gap_interval['analytics']['events']
            last_interval['analytics']['max'] = max(gap_interval['analytics']['max'], last_interval['analytics']['max'])
            last_interval['analytics']['min'] = min(gap_interval['analytics']['min'], last_interval['analytics']['min'])
            last_interval['end'] = gap_interval['end']
        else:
            counts_array.extend(counts)

    return counts_array


def calculate_monthly_peaks(counts, time_zone):
    """Given a set of count buckets and a local time zone, create monthly peak objects"""
    peaks = []

    for key, group in itertools.groupby(
        counts,
        key=lambda x: timestamp_to_local(x['timestamp'], time_zone).strftime('%Y-%m')
    ):
        peaks.append({
            'Month': f'{key}-01',
            'Peak Count': max([c['interval']['analytics']['max'] for c in group])
        })

    return peaks


def pull_space_counts(token, spaces, start, end):
    """Pull and store count buckets on the space dict"""
    for space in spaces:
        space_id = space['id']
        space_name = space['name']
        time_zone = space['time_zone']

        time_ranges = split_time_range_into_subranges_with_same_offset(
            time_zone=time_zone,
            start=start,
            end=end,
            params={'interval': '1d'}
        )

        print(f'Pulling counts for space: {space_name} from {start} to {end}')
        counts = pull_counts_for_time_ranges(token, space_id, time_ranges)

        space['counts'] = counts


def write_peaks_to_csv(spaces, start, end, peak_type, tag):
    """Given spaces w/ populated counts buckets, generate a CSV file"""
    file_name = 'density_{}_peaks{}_{}-{}.csv'.format(
        peak_type.lower(),
        f'_{tag}' if len(tag) > 0 else '',
        start.strftime(OUTPUT_DATE_FORMAT),
        end.strftime(OUTPUT_DATE_FORMAT)
    )

    field_names = ['Space', 'Date', 'Peak Count'] if peak_type == 'DAILY'\
        else ['Space', 'Month', 'Peak Count']

    outfile = open(file_name, 'w', newline='')

    writer = csv.DictWriter(outfile, fieldnames=field_names)
    writer.writeheader()

    for space in spaces:
        time_zone = space['time_zone']

        if peak_type == 'DAILY':
            for count in space['counts']:
                writer.writerow({
                    'Space': space['name'],
                    'Date': timestamp_to_local(count['timestamp'], time_zone).strftime(OUTPUT_DATE_FORMAT),
                    'Peak Count': count['interval']['analytics']['max']
                })
        else:
            for peak in calculate_monthly_peaks(space['counts'], time_zone):
                writer.writerow({
                    'Space': space['name'],
                    'Month': peak['Month'],
                    'Peak Count': peak['Peak Count']
                })

    outfile.close()


def main():
    # ====
    # parse args
    # ====
    arg_parser = argparse.ArgumentParser('Density Peaks CSV Export')
    arg_parser.add_argument(
        '-s', '--start-date',
        type=parser.parse,
        help='Start date (local) of peaks query: YYYY-MM-DD'
    )
    arg_parser.add_argument(
        '-e', '--end-date',
        type=parser.parse,
        help='End date (local) of peaks query. Format: YYYY-MM-DD'
    )
    arg_parser.add_argument(
        '-p', '--peak-type',
        type=str,
        default='DAILY',
        help='Peak type (DAILY or MONTHLY)'
    )
    arg_parser.add_argument(
        '-t', '--token',
        type=str,
        help='Density token (read-only preferred)'
    )
    arg_parser.add_argument(
        '--tag',
        type=str,
        default='',
        help='Filter Density spaces by tag name'
    )

    args = arg_parser.parse_args()

    # ensure range is inclusive of last date (evaluated at midnight)
    args.end_date += timedelta(days=1)

    # ====
    # validate args
    # ====
    if (args.start_date >= args.end_date):
      raise ValueError('Start date must be before end date!')

    if args.peak_type not in ['DAILY', 'MONTHLY']:
        raise ValueError('Peak type must be DAILY or MONTHLY')

    if args.peak_type == 'MONTHLY' and\
        (args.end_date - args.start_date < timedelta(days=28)):
        print('==! Warning: You are querying under a month of data while using the MONTHLY peak type! ==\n\n')

    # ====
    # CSV creation pipeline
    # ====
    # pull the list of spaces, optionally filtering by tag
    spaces = pull_spaces(args.token, tag=args.tag)

    # pull the counts for the time range, and attach them to the space objects
    pull_space_counts(args.token, spaces, args.start_date, args.end_date)

    # populate and save CSV data
    write_peaks_to_csv(spaces, args.start_date, args.end_date, args.peak_type, args.tag)


if __name__ == '__main__':
    main()
