#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime, timedelta
import itertools
import sys
import statistics

from dateutil import parser, tz
import pytz
import requests
import json


API_ROOT = 'https://api.density.io/v2'
OUTPUT_DATE_FORMAT = '%Y-%m-%d'


# ====
# API methods
# ====
def auth_header(token):
    """Auth header object for Density API"""
    return {'Authorization': f'Bearer {token}'}


def get_counts(token, space_id, start_time=None, end_time=None, interval='1d', paginate_next=None, order='ASC', time_segment_labels=None):
    """Convenience method to hit the space events endpoint. Will act recursively if
    data is paginated.

    Args:
        token: Density API token
        space_id: Space ID

    Kwargs:
        start_time: UTC datetime for beginning of query
        end_time: UTC datetime for end of query
        paginate_next: URL for next pagination (will override setting initial params in request)
        time_segment_labels: Label for your time segment—the hours of the day and days of the week you're scoping your data to

    Returns:
        [{...}] Counts array
    """
    if paginate_next:
        request = requests.get(paginate_next, headers=auth_header(token))
    else:
        params = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'time_segment_labels': time_segment_labels,
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
        response['results'].extend(get_counts(
            token, space_id, paginate_next=response['next']))

    return response['results']


def pull_space(token, space_id):
    """Convenience method to hit the spaces endpoint.

    Args:
        token: Density API token
        space_id: the string ID for the space you want data from

    Returns:
        {...} API space object
    """

    request = requests.get(
        f'{API_ROOT}/spaces/{space_id}',
        headers=auth_header(token)
    )
    request.raise_for_status()
    response = request.json()

    return response


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
    all_counts = []

    for subrange in time_ranges:
        counts = get_counts(
            token,
            space_id,
            start_time=subrange['start'],
            end_time=subrange['end'],
            interval='15m'
        )

        if subrange['gap'] and len(counts) > 0:
            gap_interval= counts[0]['interval']
            last_interval = all_counts[-1]['interval']

            last_interval['analytics']['entrances'] += gap_interval['analytics']['entrances']
            last_interval['analytics']['exits'] += gap_interval['analytics']['exits']
            last_interval['analytics']['events'] += gap_interval['analytics']['events']
            last_interval['analytics']['max'] = max(gap_interval['analytics']['max'], last_interval['analytics']['max'])
            last_interval['analytics']['min'] = min(gap_interval['analytics']['min'], last_interval['analytics']['min'])
            last_interval['end'] = gap_interval['end']
        else:
            all_counts += counts

    return all_counts


def pull_space_counts(token, space, start, end):
    """Pull and store count buckets on the space dict"""
    space_id = space['id']
    space_name = space['name']
    time_zone = space['time_zone']
    created_at = parser.parse(space['created_at'])

    if created_at > end.replace(tzinfo=pytz.utc):
        print(f'Skiping {space_name} because it was created too recently ({created_at})')
        return None

    # The space was created during the requested period
    if created_at > start.replace(tzinfo=pytz.utc):
        start = created_at.replace(tzinfo=None)

    time_ranges = split_time_range_into_subranges_with_same_offset(
        time_zone=time_zone,
        start=start,
        end=end,
        params={'interval': '15m'}
    )

    print(f'Pulling counts for space: {space_name} from {start} to {end}')
    return pull_counts_for_time_ranges(token, space_id, time_ranges)


def create_daily_schedule(schedule_csv, start_date, end_date):
    schedule = {}

    with open(schedule_csv, 'r') as infile:
        reader = csv.reader(infile)

        for i, row in enumerate(reader):
            if i == 0:
                continue

            weekday = int(row[2])
            class_obj = {
                'name': row[0],
                'start_time': parser.parse(row[3]).time(),
                'end_time': parser.parse(row[4]).time(),
            }

            if not weekday in schedule:
                schedule[weekday] = [class_obj]
            else:
                schedule[weekday].append(class_obj)

    day = start_date
    daily_schedule = []

    while day < end_date:
        classes = schedule[day.weekday()]
        for c in classes:
            daily_schedule.append({
                'start_time': day.replace(hour=c['start_time'].hour, minute=c['start_time'].minute),
                'end_time': day.replace(hour=c['end_time'].hour, minute=c['end_time'].minute),
                'name': c['name'],
            })


        day += timedelta(days=1)

    return daily_schedule


def add_peak_counts_to_daily_schedule(daily_schedule, local_tz, counts):
    for c in daily_schedule:
        start_local = c['start_time'].replace(tzinfo=tz.gettz(local_tz))
        end_local = c['end_time'].replace(tzinfo=tz.gettz(local_tz))

        relevant_buckets = list(filter(
            lambda x: timestamp_to_local(x['interval']['start'], local_tz) >= start_local and timestamp_to_local(x['interval']['end'], local_tz) < end_local,
            counts
        ))
        c['peak_occupancy'] = max([b['interval']['analytics']['max'] for b in relevant_buckets])


def write_schedule_peaks_to_csv(space, start, end, daily_schedule):
    """Given daily schedule w/ populated peak_counts, generate a CSV file"""
    file_name = 'density_schedule_peaks_{}-{}.csv'.format(
        start.strftime(OUTPUT_DATE_FORMAT),
        end.strftime(OUTPUT_DATE_FORMAT)
    )

    field_names = ['Space', 'Class Name', 'Class Start', 'Class End', 'Peak Occupancy']

    outfile = open(file_name, 'w', newline='')

    writer = csv.DictWriter(outfile, fieldnames=field_names)
    writer.writeheader()

    for c in daily_schedule:
        writer.writerow({
            'Space': space['name'],
            'Class Name': c['name'],
            'Class Start': c['start_time'],
            'Class End': c['end_time'],
            'Peak Occupancy': c['peak_occupancy'],
        })

    outfile.close()


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-s', '--start-date',
        type=parser.parse,
        required=True,
        help='Start date (local) of peaks query: YYYY-MM-DD'
    )
    arg_parser.add_argument(
        '-e', '--end-date',
        type=parser.parse,
        required=True,
        help='End date (local) of peaks query. Format: YYYY-MM-DD'
    )
    arg_parser.add_argument(
        '-t', '--token',
        type=str,
        required=True,
        help='Density API token (read-only preferred)'
    )
    arg_parser.add_argument(
        '-sch', '--schedule',
        type=str,
        required=True,
        help='Path to the schedule csv file'
    )
    arg_parser.add_argument(
        '-sid', '--space-id',
        type=str,
        required=True,
        help='Space ID to pull data from'
    )

    args = arg_parser.parse_args()

    # ensure range is inclusive of last date (evaluated at midnight)
    args.end_date += timedelta(days=1)

    # validation / sanity checks
    if args.start_date >= args.end_date:
        raise ValueError('Start date must be before end date!')

    return args


def create_csv(parsed_args):
    # craft the daily schedule based on the passed through schedule csv
    # over the dates specified
    daily_schedule = create_daily_schedule(
        parsed_args.schedule,
        parsed_args.start_date,
        parsed_args.end_date
    )

    # pull the list of spaces, optionally filtering by tag
    space = pull_space(parsed_args.token, parsed_args.space_id)

    # pull the counts for the time range, and attach them to the space objects
    counts = pull_space_counts(parsed_args.token, space, parsed_args.start_date, parsed_args.end_date)

    add_peak_counts_to_daily_schedule(daily_schedule, space['time_zone'], counts)

    # populate and save CSV data
    write_schedule_peaks_to_csv(space, parsed_args.start_date, parsed_args.end_date, daily_schedule)



if __name__ == '__main__':
    create_csv(parse_args())
