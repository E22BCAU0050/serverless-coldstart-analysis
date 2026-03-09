#!/usr/bin/env python3
"""
Cold Start Dataset Extractor
============================
Account : 873166938412
Region  : us-east-1
Project : coldstart-research

Pulls metrics from:
  1. DynamoDB  coldstart-research-metrics  (application-level telemetry)
  2. CloudWatch Logs  REPORT lines         (InitDuration, Duration, MemoryUsed)
  3. CloudWatch Metrics API                (p95, p99, invocation counts)

Outputs: cold_start_dataset.csv  (ready for ML/DL training)

Usage:
  pip install boto3 pandas tqdm
  python extract_dataset.py [--days 7] [--output cold_start_dataset.csv]
"""

import re
import sys
import time
import json
import argparse
import boto3
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────
AWS_ACCOUNT  = "873166938412"
AWS_REGION   = "us-east-1"
PROJECT      = "coldstart-research"

FUNCTIONS = {
    "non-vpc":     f"{PROJECT}-api-non-vpc",
    "vpc":         f"{PROJECT}-api-vpc",
    "provisioned": f"{PROJECT}-api-provisioned",
}
TABLE_METRICS = f"{PROJECT}-metrics"
TABLE_BOOKS   = f"{PROJECT}-books"

parser = argparse.ArgumentParser(description='Extract cold start dataset')
parser.add_argument('--days',   type=int, default=7,                     help='How many days back to query')
parser.add_argument('--output', default='cold_start_dataset.csv',         help='Output CSV filename')
args = parser.parse_args()

# ── AWS clients ───────────────────────────────────────────────────────────────
dynamo     = boto3.resource('dynamodb', region_name=AWS_REGION)
logs       = boto3.client('logs',       region_name=AWS_REGION)
cw         = boto3.client('cloudwatch', region_name=AWS_REGION)

END_TIME   = datetime.now(timezone.utc)
START_TIME = END_TIME - timedelta(days=args.days)

# ── REPORT line regex ──────────────────────────────────────────────────────────
REPORT_RE = re.compile(
    r'REPORT RequestId:\s*(?P<request_id>[\w-]+)\s+'
    r'Duration:\s*(?P<duration>[\d.]+) ms\s+'
    r'Billed Duration:\s*(?P<billed>[\d.]+) ms\s+'
    r'Memory Size:\s*(?P<memory>\d+) MB\s+'
    r'Max Memory Used:\s*(?P<max_mem>\d+) MB'
    r'(?:\s+Init Duration:\s*(?P<init>[\d.]+) ms)?'
)

# =============================================================================
# 1. Pull from DynamoDB metrics table
# =============================================================================
def pull_dynamo_metrics():
    print(f"\n[1/3] Scanning DynamoDB: {TABLE_METRICS} ...")
    table = dynamo.Table(TABLE_METRICS)
    items = []
    kwargs = {}
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp['Items'])
        sys.stdout.write(f"  fetched {len(items)} records...\r")
        sys.stdout.flush()
        if 'LastEvaluatedKey' not in resp:
            break
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']
    print(f"  → {len(items)} records from DynamoDB metrics table")
    return items

# =============================================================================
# 2. Pull REPORT lines from CloudWatch Logs Insights
# =============================================================================
def query_cw_logs(log_group: str, func_type: str) -> list[dict]:
    query = """
    fields @timestamp, @message
    | filter @message like /REPORT RequestId/
    | sort @timestamp asc
    | limit 10000
    """
    try:
        resp = logs.start_query(
            logGroupName=log_group,
            startTime=int(START_TIME.timestamp()),
            endTime=int(END_TIME.timestamp()),
            queryString=query
        )
        qid = resp['queryId']
        for _ in range(40):
            time.sleep(3)
            result = logs.get_query_results(queryId=qid)
            if result['status'] == 'Complete':
                return result['results']
            elif result['status'] in ('Failed', 'Cancelled'):
                print(f"    ⚠ Query {result['status']}")
                return []
        print("    ⚠ Query timed out")
        return []
    except logs.exceptions.ResourceNotFoundException:
        print(f"    ⚠ Log group not found: {log_group}")
        return []
    except Exception as e:
        print(f"    ⚠ Error: {e}")
        return []

def parse_report(msg: str, ts: str, func_type: str) -> dict | None:
    m = REPORT_RE.search(msg)
    if not m:
        return None
    d = m.groupdict()
    return {
        'source':          'cloudwatch_logs',
        'request_id':      d['request_id'],
        'timestamp':       ts,
        'function_type':   func_type,
        'duration_ms':     float(d['duration']),
        'billed_ms':       float(d['billed']),
        'memory_size_mb':  int(d['memory']),
        'max_memory_mb':   int(d['max_mem']),
        'init_duration_ms': float(d['init']) if d['init'] else 0.0,
        'is_cold_start':   d['init'] is not None,
    }

def pull_cw_logs_all() -> list[dict]:
    print(f"\n[2/3] Querying CloudWatch Logs for REPORT lines ...")
    records = []
    for func_type, func_name in FUNCTIONS.items():
        log_group = f"/aws/lambda/{func_name}"
        print(f"  Querying {log_group} ...")
        rows = query_cw_logs(log_group, func_type)
        parsed = []
        for row in rows:
            msg = next((f['value'] for f in row if f['field'] == '@message'), '')
            ts  = next((f['value'] for f in row if f['field'] == '@timestamp'), '')
            p = parse_report(msg, ts, func_type)
            if p:
                parsed.append(p)
        print(f"  → {len(parsed)} REPORT lines from {func_type}")
        records.extend(parsed)
    return records

# =============================================================================
# 3. Pull CloudWatch aggregate metrics per function
# =============================================================================
def get_cw_stat(func_name: str, metric: str, stat: str, extended=False) -> float | None:
    try:
        kwargs = dict(
            Namespace='AWS/Lambda',
            MetricName=metric,
            Dimensions=[{'Name': 'FunctionName', 'Value': func_name}],
            StartTime=START_TIME,
            EndTime=END_TIME,
            Period=3600,
        )
        if extended:
            kwargs['ExtendedStatistics'] = [stat]
        else:
            kwargs['Statistics'] = [stat]
        resp = cw.get_metric_statistics(**kwargs)
        dp = resp['Datapoints']
        if not dp:
            return None
        vals = [d.get(stat) or d.get('ExtendedStatistics', {}).get(stat) for d in dp]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    except Exception:
        return None

def pull_cw_metrics_all() -> dict:
    print(f"\n[3/3] Pulling CloudWatch aggregate metrics per function ...")
    agg = {}
    for func_type, func_name in FUNCTIONS.items():
        print(f"  Pulling CW metrics for {func_name} ...")
        agg[func_type] = {
            'cw_avg_duration_ms':      get_cw_stat(func_name, 'Duration',            'Average'),
            'cw_p95_duration_ms':      get_cw_stat(func_name, 'Duration',            'p95', extended=True),
            'cw_p99_duration_ms':      get_cw_stat(func_name, 'Duration',            'p99', extended=True),
            'cw_avg_init_ms':          get_cw_stat(func_name, 'InitDuration',         'Average'),
            'cw_max_init_ms':          get_cw_stat(func_name, 'InitDuration',         'Maximum'),
            'cw_total_invocations':    get_cw_stat(func_name, 'Invocations',          'Sum'),
            'cw_total_errors':         get_cw_stat(func_name, 'Errors',              'Sum'),
            'cw_total_throttles':      get_cw_stat(func_name, 'Throttles',           'Sum'),
            'cw_max_concurrent_execs': get_cw_stat(func_name, 'ConcurrentExecutions','Maximum'),
        }
        for k, v in agg[func_type].items():
            print(f"    {k}: {v}")
    return agg

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("COLD START DATASET EXTRACTOR")
    print(f"Account : {AWS_ACCOUNT}")
    print(f"Region  : {AWS_REGION}")
    print(f"Range   : {START_TIME.date()} → {END_TIME.date()} ({args.days} days)")
    print(f"Output  : {args.output}")
    print("=" * 60)

    # 1. DynamoDB
    dynamo_items = pull_dynamo_metrics()
    dynamo_map   = {item['requestId']: item for item in dynamo_items}

    # 2. CloudWatch Logs
    cw_records = pull_cw_logs_all()

    # 3. Enrich CW records with DynamoDB app data
    cw_ids = set()
    enriched = []
    for rec in cw_records:
        rid  = rec.get('request_id', '')
        cw_ids.add(rid)
        dyn  = dynamo_map.get(rid, {})
        enriched.append({
            **rec,
            'app_duration_ms': float(dyn.get('durationMs', 0) or 0),
            'app_is_cold':     bool(dyn.get('isColdStart', False)),
            'api_path':        str(dyn.get('path', '')),
            'api_method':      str(dyn.get('method', '')),
        })

    # 4. Add DynamoDB-only records (not yet in CW logs)
    for item in dynamo_items:
        if item['requestId'] not in cw_ids:
            enriched.append({
                'source':          'dynamodb',
                'request_id':      item['requestId'],
                'timestamp':       str(item.get('timestamp', '')),
                'function_type':   str(item.get('functionType', 'unknown')),
                'duration_ms':     float(item.get('durationMs', 0) or 0),
                'billed_ms':       0,
                'memory_size_mb':  int(item.get('memoryMB', 128) or 128),
                'max_memory_mb':   0,
                'init_duration_ms': 0,
                'is_cold_start':   bool(item.get('isColdStart', False)),
                'app_duration_ms': float(item.get('durationMs', 0) or 0),
                'app_is_cold':     bool(item.get('isColdStart', False)),
                'api_path':        str(item.get('path', '')),
                'api_method':      str(item.get('method', '')),
            })

    if not enriched:
        print("\n⚠ No records found. Make sure you have invoked the Lambda functions.")
        print("  Run:  ./run_all.sh test   then wait a few minutes before extracting.")
        sys.exit(0)

    # 5. Build DataFrame + feature engineering
    df = pd.DataFrame(enriched)
    df['timestamp']        = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['hour_of_day']      = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week']      = df['timestamp'].dt.dayofweek.fillna(0).astype(int)
    df['vpc_flag']         = (df['function_type'] == 'vpc').astype(int)
    df['provisioned_flag'] = (df['function_type'] == 'provisioned').astype(int)
    df['container_flag']   = 0   # Update to 1 when container-image Lambda added
    df['cold_start_flag']  = df['is_cold_start'].astype(int)
    df['memory_size_mb']   = df['memory_size_mb'].fillna(128).astype(int)

    # 6. Attach CW aggregate stats
    cw_agg = pull_cw_metrics_all()
    for col in ['cw_avg_duration_ms','cw_p95_duration_ms','cw_p99_duration_ms',
                'cw_avg_init_ms','cw_max_init_ms','cw_total_invocations',
                'cw_total_errors','cw_max_concurrent_execs']:
        df[col] = df['function_type'].map(lambda ft: cw_agg.get(ft, {}).get(col))

    # 7. Final column order (ML-ready)
    COLS = [
        'request_id', 'timestamp', 'function_type', 'source',
        # Features
        'memory_size_mb', 'vpc_flag', 'provisioned_flag', 'container_flag',
        'hour_of_day', 'day_of_week', 'api_path', 'api_method',
        # Targets
        'init_duration_ms', 'duration_ms', 'cold_start_flag',
        # Auxiliary
        'billed_ms', 'max_memory_mb', 'app_duration_ms',
        # CW aggregate
        'cw_avg_duration_ms', 'cw_p95_duration_ms', 'cw_p99_duration_ms',
        'cw_avg_init_ms', 'cw_max_init_ms',
        'cw_total_invocations', 'cw_total_errors', 'cw_max_concurrent_execs',
    ]
    existing = [c for c in COLS if c in df.columns]
    df = df[existing].drop_duplicates(subset=['request_id'], keep='last')
    df = df.sort_values('timestamp', ascending=True)

    df.to_csv(args.output, index=False)

    # 8. Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total records  : {len(df)}")
    print(f"  Cold starts    : {df['cold_start_flag'].sum()}")
    print(f"  Warm starts    : {(df['cold_start_flag'] == 0).sum()}")
    print(f"  Output file    : {args.output}")
    print(f"  Columns        : {len(df.columns)}")
    print()
    print("  Records per function:")
    print(df['function_type'].value_counts().to_string())
    cold = df[df['cold_start_flag'] == 1]
    if len(cold) > 0:
        print("\n  Cold Start Init Duration (ms):")
        print(cold.groupby('function_type')['init_duration_ms']
              .describe()[['count','mean','std','min','50%','95%','max']]
              .round(2).to_string())
    print("=" * 60)

if __name__ == '__main__':
    main()
