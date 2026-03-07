#!/usr/bin/env python3
"""
Cold Start Dataset Extractor
============================
Pulls cold start metrics from:
  1. CloudWatch Logs — REPORT lines (InitDuration, Duration, MemoryUsed)
  2. DynamoDB Metrics Table — application-level telemetry
  3. CloudWatch Metrics API — Lambda Insights data

Outputs a structured CSV ready for ML/DL model training.

Usage:
    pip install boto3 pandas tqdm
    python extract_dataset.py --region us-east-1 --project coldstart-research --days 7
"""

import boto3
import pandas as pd
import json
import re
import time
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Extract cold start dataset')
parser.add_argument('--region',  default='us-east-1')
parser.add_argument('--project', default='coldstart-research')
parser.add_argument('--days',    type=int, default=7)
parser.add_argument('--output',  default='cold_start_dataset.csv')
args = parser.parse_args()

REGION       = args.region
PROJECT      = args.project
DAYS         = args.days
OUTPUT_FILE  = args.output

# Lambda function names
FUNCTIONS = {
    'non-vpc':    f'{PROJECT}-api-non-vpc',
    'vpc':        f'{PROJECT}-api-vpc',
    'provisioned':f'{PROJECT}-api-provisioned',
}

METRICS_TABLE = f'{PROJECT}-metrics'

logs_client = boto3.client('logs',       region_name=REGION)
dynamo      = boto3.resource('dynamodb', region_name=REGION)
cw_client   = boto3.client('cloudwatch', region_name=REGION)

# ─────────────────────────────────────────────────────────────
# 1. Parse REPORT lines from CloudWatch Logs
# ─────────────────────────────────────────────────────────────

REPORT_RE = re.compile(
    r'REPORT RequestId: (?P<request_id>[\w-]+)\s+'
    r'Duration: (?P<duration>[\d.]+) ms\s+'
    r'Billed Duration: (?P<billed_duration>[\d.]+) ms\s+'
    r'Memory Size: (?P<memory_size>\d+) MB\s+'
    r'Max Memory Used: (?P<max_memory_used>\d+) MB'
    r'(?:\s+Init Duration: (?P<init_duration>[\d.]+) ms)?'
)

def query_logs(log_group: str, start_time: datetime, end_time: datetime) -> list[dict]:
    """Query CW Logs Insights for REPORT lines."""
    query = """
    fields @timestamp, @message
    | filter @message like /REPORT RequestId/
    | sort @timestamp asc
    | limit 10000
    """
    start_ms = int(start_time.timestamp())
    end_ms   = int(end_time.timestamp())

    response = logs_client.start_query(
        logGroupName=log_group,
        startTime=start_ms,
        endTime=end_ms,
        queryString=query
    )
    query_id = response['queryId']
    print(f"  Started CW Logs query: {query_id}")

    # Poll until complete
    for _ in range(30):
        time.sleep(3)
        result = logs_client.get_query_results(queryId=query_id)
        if result['status'] == 'Complete':
            return result['results']
        elif result['status'] in ['Failed', 'Cancelled']:
            print(f"  Query {query_id} {result['status']}")
            return []

    print("  Query timed out")
    return []

def parse_report_line(message: str, timestamp: str, function_type: str) -> dict | None:
    m = REPORT_RE.search(message)
    if not m:
        return None
    d = m.groupdict()
    return {
        'request_id':      d['request_id'],
        'timestamp':       timestamp,
        'function_type':   function_type,
        'duration_ms':     float(d['duration']),
        'billed_ms':       float(d['billed_duration']),
        'memory_size_mb':  int(d['memory_size']),
        'max_memory_mb':   int(d['max_memory_used']),
        'init_duration_ms': float(d['init_duration']) if d['init_duration'] else 0.0,
        'is_cold_start':   d['init_duration'] is not None,
    }

# ─────────────────────────────────────────────────────────────
# 2. Pull from DynamoDB Metrics Table
# ─────────────────────────────────────────────────────────────

def pull_dynamo_metrics() -> list[dict]:
    table = dynamo.Table(METRICS_TABLE)
    items = []
    kwargs = {}
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp['Items'])
        if 'LastEvaluatedKey' not in resp:
            break
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']
    return items

# ─────────────────────────────────────────────────────────────
# 3. Pull Lambda CW Metrics (p95, p99 per function)
# ─────────────────────────────────────────────────────────────

def pull_cw_metrics(function_name: str, start: datetime, end: datetime) -> dict:
    def get_stat(metric_name: str, stat: str, extended: bool = False):
        kwargs = dict(
            Namespace='AWS/Lambda',
            MetricName=metric_name,
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start,
            EndTime=end,
            Period=3600,
        )
        if extended:
            kwargs['ExtendedStatistics'] = [stat]
        else:
            kwargs['Statistics'] = [stat]
        resp = cw_client.get_metric_statistics(**kwargs)
        dp = resp['Datapoints']
        if not dp:
            return None
        vals = [d.get(stat) or d.get('ExtendedStatistics', {}).get(stat) for d in dp]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        'cw_avg_duration':    get_stat('Duration', 'Average'),
        'cw_p95_duration':    get_stat('Duration', 'p95', extended=True),
        'cw_p99_duration':    get_stat('Duration', 'p99', extended=True),
        'cw_avg_init':        get_stat('InitDuration', 'Average'),
        'cw_invocations':     get_stat('Invocations', 'Sum'),
        'cw_errors':          get_stat('Errors', 'Sum'),
        'cw_throttles':       get_stat('Throttles', 'Sum'),
        'cw_concurrent_execs':get_stat('ConcurrentExecutions', 'Maximum'),
    }

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    end_time   = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_records = []

    # ── Step 1: CloudWatch Logs ───────────────────────────────
    print("\n[1/3] Extracting CloudWatch REPORT lines...")
    for func_type, func_name in FUNCTIONS.items():
        log_group = f'/aws/lambda/{func_name}'
        print(f"  Querying {log_group}...")
        try:
            results = query_logs(log_group, start_time, end_time)
            for row in results:
                msg = next((f['value'] for f in row if f['field'] == '@message'), '')
                ts  = next((f['value'] for f in row if f['field'] == '@timestamp'), '')
                parsed = parse_report_line(msg, ts, func_type)
                if parsed:
                    all_records.append(parsed)
            print(f"    → {len([r for r in all_records if r['function_type'] == func_type])} records")
        except Exception as e:
            print(f"    ⚠ Error querying {log_group}: {e}")

    # ── Step 2: DynamoDB ──────────────────────────────────────
    print("\n[2/3] Pulling DynamoDB application metrics...")
    try:
        dynamo_items = pull_dynamo_metrics()
        print(f"  → {len(dynamo_items)} records from DynamoDB metrics table")

        # Enrich CW records with DynamoDB data
        dynamo_map = {item['requestId']: item for item in dynamo_items}
        for rec in all_records:
            dyn = dynamo_map.get(rec.get('request_id'), {})
            rec['app_duration_ms']  = float(dyn.get('durationMs', 0) or 0)
            rec['app_is_cold']      = bool(dyn.get('isColdStart', False))
            rec['api_path']         = dyn.get('path', '')
            rec['api_method']       = dyn.get('method', '')

        # Add standalone DynamoDB records not in CW logs
        cw_ids = {r['request_id'] for r in all_records}
        for item in dynamo_items:
            if item['requestId'] not in cw_ids:
                all_records.append({
                    'request_id':       item['requestId'],
                    'timestamp':        str(item.get('timestamp', '')),
                    'function_type':    str(item.get('functionType', 'unknown')),
                    'duration_ms':      float(item.get('durationMs', 0) or 0),
                    'billed_ms':        0,
                    'memory_size_mb':   int(item.get('memorySize', 0) or 0),
                    'max_memory_mb':    0,
                    'init_duration_ms': 0,
                    'is_cold_start':    bool(item.get('isColdStart', False)),
                    'app_duration_ms':  float(item.get('durationMs', 0) or 0),
                    'app_is_cold':      bool(item.get('isColdStart', False)),
                    'api_path':         str(item.get('path', '')),
                    'api_method':       str(item.get('method', '')),
                })
    except Exception as e:
        print(f"  ⚠ DynamoDB error: {e}")

    # ── Step 3: CloudWatch Metrics ────────────────────────────
    print("\n[3/3] Pulling CloudWatch aggregate metrics per function...")
    cw_agg = {}
    for func_type, func_name in FUNCTIONS.items():
        try:
            cw_agg[func_type] = pull_cw_metrics(func_name, start_time, end_time)
        except Exception as e:
            print(f"  ⚠ CW metrics error for {func_name}: {e}")
            cw_agg[func_type] = {}

    # Attach CW aggregate stats to each record
    for rec in all_records:
        stats = cw_agg.get(rec.get('function_type'), {})
        rec.update(stats)

    # ── Build DataFrame ───────────────────────────────────────
    if not all_records:
        print("\n⚠ No records collected. Make sure you have invoked the Lambda functions first.")
        return

    df = pd.DataFrame(all_records)

    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['hour_of_day']    = df['timestamp'].dt.hour
    df['day_of_week']    = df['timestamp'].dt.dayofweek
    df['vpc_flag']       = (df['function_type'] == 'vpc').astype(int)
    df['provisioned_flag'] = (df['function_type'] == 'provisioned').astype(int)
    df['container_flag'] = 0  # Set to 1 when container image Lambda is deployed
    df['cold_start_flag'] = df['is_cold_start'].astype(int)

    # Select and order final ML features
    ml_cols = [
        'request_id', 'timestamp', 'function_type',
        'memory_size_mb', 'vpc_flag', 'provisioned_flag', 'container_flag',
        'hour_of_day', 'day_of_week',
        'init_duration_ms',        # TARGET: cold start latency
        'duration_ms',             # TARGET: execution latency
        'billed_ms', 'max_memory_mb',
        'cold_start_flag',
        'api_path', 'api_method',
        'cw_avg_duration', 'cw_p95_duration', 'cw_p99_duration',
        'cw_avg_init', 'cw_invocations', 'cw_errors', 'cw_concurrent_execs',
    ]
    existing_cols = [c for c in ml_cols if c in df.columns]
    df = df[existing_cols].drop_duplicates(subset=['request_id'], keep='last')

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Dataset saved: {OUTPUT_FILE}")
    print(f"   Total records:   {len(df)}")
    print(f"   Cold starts:     {df['cold_start_flag'].sum() if 'cold_start_flag' in df.columns else 'N/A'}")
    print(f"   Functions:")
    if 'function_type' in df.columns:
        print(df['function_type'].value_counts().to_string())
    print(f"\n   Columns: {list(df.columns)}")
    if 'init_duration_ms' in df.columns:
        cold_df = df[df['cold_start_flag'] == 1]
        if len(cold_df) > 0:
            print(f"\n   Cold Start Stats:")
            print(cold_df.groupby('function_type')['init_duration_ms'].describe().round(2).to_string())

if __name__ == '__main__':
    main()
