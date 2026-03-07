# Cold Start Latency Research — AWS Lambda
### Hybrid ML–DL Framework for Serverless Deployment Optimization

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  API Gateway (Regional)                                             │
│    /health   /books   /books/{bookId}                               │
└─────────────┬──────────────────────────────────────────────────────┘
              │ invokes
    ┌─────────┼──────────────────────────────────────┐
    ▼         ▼                                      ▼
┌─────────┐ ┌─────────────────┐       ┌───────────────────────────┐
│ Lambda  │ │ Lambda (VPC)    │       │ Lambda (Provisioned CC)   │
│ Non-VPC │ │ Private Subnets │       │ Alias: live               │
│ ZIP pkg │ │ NAT Gateway     │       │ 2 warm instances          │
└────┬────┘ └──────┬──────────┘       └────────────┬──────────────┘
     │             │                               │
     └─────────────┴───────────────────────────────┘
                   │ reads/writes
        ┌──────────┴──────────┐
        │    DynamoDB          │
        │  ┌────────────────┐ │
        │  │ books table    │ │  ← Production-like inventory data (20 books seeded)
        │  │ metrics table  │ │  ← Cold start telemetry (auto-expires in 30 days)
        │  └────────────────┘ │
        └─────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  CloudWatch Logs    │
        │  Lambda Insights    │
        │  CW Dashboard       │
        └─────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  Data Pipeline      │
        │  extract_dataset.py │ ← Exports CSV for ML training
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │  ML + DL Models     │
        │  train_models.py    │ ← Random Forest + LSTM + Recommender
        └─────────────────────┘
```

---

## Step-by-Step Setup

### Prerequisites
```bash
# Install AWS CLI
pip install awscli
aws configure   # Enter your AWS credentials

# Install tools
npm install -g k6
pip install boto3 pandas scikit-learn xgboost tensorflow matplotlib tqdm
```

### Step 1 — Deploy Infrastructure

```bash
aws cloudformation deploy \
  --template-file cloudformation/main-stack.yaml \
  --stack-name coldstart-research \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName=coldstart-research \
    Environment=dev \
    LambdaMemorySize=512 \
    EnableProvisionedConcurrency=false \
  --region us-east-1
```

> **Cost note:** The NAT Gateway (~$0.045/hr) is the biggest cost driver.
> Delete the stack when not actively testing to avoid charges.

### Step 2 — Get the API URL

```bash
aws cloudformation describe-stacks \
  --stack-name coldstart-research \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiBaseUrl`].OutputValue' \
  --output text
```

### Step 3 — Seed the Database

```bash
# Seed with 20 production-like book records
aws lambda invoke \
  --function-name coldstart-research-seeder \
  --payload '{"action": "seed"}' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
```

### Step 4 — Test the API manually

```bash
API_URL="https://YOUR_ID.execute-api.us-east-1.amazonaws.com/dev"

# Health check
curl $API_URL/health

# List books
curl $API_URL/books

# Get books by genre
curl "$API_URL/books?genre=Technology"

# Create a book
curl -X POST $API_URL/books \
  -H "Content-Type: application/json" \
  -d '{"title":"My Book","author":"Test","genre":"Technology","publishedYear":2024,"price":29.99,"stock":100}'
```

### Step 5 — Run Load Tests

```bash
# Cold start burst test (20 concurrent VUs, 1 iteration each — forces cold starts)
k6 run --env API_URL=$API_URL load-testing/load-test.js

# Custom: Force cold starts by waiting 15 min between runs
# Lambda instances are recycled after ~15 min of inactivity
```

### Step 6 — Compare VPC vs Non-VPC

Invoke the VPC function directly to compare cold start:

```bash
# Direct Lambda invocation (bypasses API GW for raw timing)
time aws lambda invoke \
  --function-name coldstart-research-api-non-vpc \
  --payload '{"path":"/health","httpMethod":"GET"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/out.json

time aws lambda invoke \
  --function-name coldstart-research-api-vpc \
  --payload '{"path":"/health","httpMethod":"GET"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/out.json
```

### Step 7 — Extract Dataset

```bash
python data-pipeline/extract_dataset.py \
  --region us-east-1 \
  --project coldstart-research \
  --days 7 \
  --output cold_start_dataset.csv
```

### Step 8 — Train Models

```bash
python ml-models/train_models.py \
  --dataset cold_start_dataset.csv \
  --output_dir model_outputs
```

---

## Where Metrics Are Stored

| Storage | What's stored | How to access |
|---------|--------------|---------------|
| **CloudWatch Logs** | Raw REPORT lines with `InitDuration`, `Duration`, `MemoryUsed` | AWS Console → CloudWatch → Log Groups → `/aws/lambda/coldstart-research-*` |
| **Lambda Insights** | CPU, memory, network, init duration per invocation | AWS Console → CloudWatch → Lambda Insights |
| **DynamoDB `metrics` table** | App-level telemetry: request ID, cold start flag, duration, function type | AWS Console → DynamoDB → Tables → `coldstart-research-metrics` |
| **CloudWatch Dashboard** | Visual dashboard for all 3 functions | See `Outputs.CloudWatchDashboard` URL in stack outputs |

---

## Reading Cold Start from CloudWatch Logs

A cold start shows `Init Duration` in the REPORT line:
```
REPORT RequestId: abc-123  Duration: 245.32 ms  Billed: 246 ms  
Memory: 512 MB  Max Used: 88 MB  Init Duration: 812.45 ms
```
A **warm invocation** has no `Init Duration`:
```
REPORT RequestId: xyz-789  Duration: 18.44 ms  Billed: 19 ms  
Memory: 512 MB  Max Used: 88 MB
```

---

## Dataset Schema (for ML/DL)

| Column | Description | Type |
|--------|-------------|------|
| `memory_size_mb` | Lambda memory config | int |
| `vpc_flag` | 1 = inside VPC, 0 = outside | int |
| `provisioned_flag` | 1 = provisioned concurrency | int |
| `container_flag` | 1 = container image, 0 = ZIP | int |
| `hour_of_day` | 0–23, captures diurnal patterns | int |
| `day_of_week` | 0=Mon … 6=Sun | int |
| `init_duration_ms` | **TARGET**: cold start latency | float |
| `duration_ms` | Execution latency | float |
| `cold_start_flag` | 1 = cold start invocation | int |
| `cw_p95_duration` | P95 latency from CloudWatch | float |
| `cw_p99_duration` | P99 latency from CloudWatch | float |
| `cw_concurrent_execs` | Peak concurrent executions | float |

---

## Cost Estimate (College Project)

| Resource | Monthly est. (light usage) |
|----------|--------------------------|
| Lambda (3 functions × 10k invocations) | ~$0.00 (free tier) |
| DynamoDB (on-demand, ~100 writes/day) | ~$0.01 |
| API Gateway (10k requests) | ~$0.04 |
| CloudWatch Logs (14-day retention) | ~$0.10 |
| NAT Gateway | ~$1.50/day ← **Delete when not testing!** |
| **Total active testing session (3 hrs)** | **~$0.20** |

> 💡 To minimize NAT Gateway costs, run your experiments in focused sessions
> and delete the stack between sessions: `aws cloudformation delete-stack --stack-name coldstart-research`

---

## Experiment Methodology for Paper

### Controlled Variables
- Runtime: Python 3.12 (same across all functions)
- Handler code: Identical logic across all variants
- DynamoDB: Same table, same data

### Independent Variables
- VPC vs non-VPC
- Memory size: 128MB / 256MB / 512MB / 1024MB / 2048MB
- Deployment package: ZIP (current) vs Container Image
- Provisioned Concurrency: enabled vs disabled

### Dependent Variables
- `init_duration_ms` — cold start latency
- `duration_ms` — execution latency
- p95 / p99 latency
- Cost per 10k invocations

### Recommended Experiment Runs
1. **Idle test**: Wait 15+ mins, invoke once → record cold start
2. **Burst test**: 20 concurrent invocations → count cold starts
3. **Memory scaling**: Deploy 5 variants (128–2048MB), compare init_duration
4. **VPC comparison**: Same function, VPC vs non-VPC
5. **Provisioned CC**: Enable on one function, compare to baseline
