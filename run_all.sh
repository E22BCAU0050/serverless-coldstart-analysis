#!/usr/bin/env bash
# =============================================================================
# Cold Start Research — MASTER SCRIPT
# Account : 873166938412
# Region  : us-east-1
# Memory  : 128 MB
# OS      : macOS / Linux
#
# USAGE:
#   chmod +x run_all.sh
#   ./run_all.sh deploy       ← Step 1: Deploy CloudFormation stack
#   ./run_all.sh seed         ← Step 2: Seed DB with 20 books
#   ./run_all.sh test         ← Step 3: Run cold start experiment (curl-based)
#   ./run_all.sh load         ← Step 4: Run k6 load test (needs k6 installed)
#   ./run_all.sh extract      ← Step 5: Export metrics from DynamoDB → CSV
#   ./run_all.sh dashboard    ← Open CloudWatch dashboard in browser
#   ./run_all.sh destroy      ← Teardown (saves cost!)
#   ./run_all.sh status       ← Check stack status
# =============================================================================

set -euo pipefail

# ── Config (hardcoded for your account) ──────────────────────────────────────
AWS_ACCOUNT_ID="873166938412"
AWS_REGION="us-east-1"
STACK_NAME="coldstart-research"
PROJECT="coldstart-research"
MEMORY_MB="128"

FUNC_NON_VPC="${PROJECT}-api-non-vpc"
FUNC_VPC="${PROJECT}-api-vpc"
FUNC_PROVISIONED="${PROJECT}-api-provisioned"
FUNC_SEEDER="${PROJECT}-seeder"
TABLE_BOOKS="${PROJECT}-books"
TABLE_METRICS="${PROJECT}-metrics"
CFN_TEMPLATE="$(dirname "$0")/main-stack.yaml"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
info() { echo -e "${BLUE}[→]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[✗]${NC} $*" && exit 1; }
banner() {
  echo ""
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  $*${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ── Helper: get API URL from stack outputs ────────────────────────────────────
get_api_url() {
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiBaseUrl`].OutputValue' \
    --output text 2>/dev/null
}

# ── Helper: invoke Lambda directly and return log ─────────────────────────────
invoke_lambda() {
  local func="$1"
  local payload="$2"
  local outfile="/tmp/lambda_out_$$.json"
  aws lambda invoke \
    --function-name "$func" \
    --payload "$payload" \
    --cli-binary-format raw-in-base64-out \
    --log-type Tail \
    --region "$AWS_REGION" \
    --query 'LogResult' \
    --output text \
    "$outfile" 2>/dev/null | base64 --decode 2>/dev/null || true
  cat "$outfile" 2>/dev/null && rm -f "$outfile"
}

# =============================================================================
# COMMAND: deploy
# =============================================================================
cmd_deploy() {
  banner "STEP 1 — Deploying CloudFormation Stack"
  info "Stack    : $STACK_NAME"
  info "Account  : $AWS_ACCOUNT_ID"
  info "Region   : $AWS_REGION"
  info "Template : $CFN_TEMPLATE"
  echo ""

  [[ -f "$CFN_TEMPLATE" ]] || fail "Template not found: $CFN_TEMPLATE"

  # Verify AWS identity
  info "Verifying AWS credentials..."
  CALLER=$(aws sts get-caller-identity \
    --query '{Account:Account,Arn:Arn}' \
    --output json --region "$AWS_REGION")
  echo "$CALLER" | python3 -m json.tool 2>/dev/null || echo "$CALLER"
  echo ""

  info "Deploying stack (this takes ~5-8 minutes for NAT Gateway)..."
  aws cloudformation deploy \
    --template-file "$CFN_TEMPLATE" \
    --stack-name "$STACK_NAME" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$AWS_REGION" \
    --no-fail-on-empty-changeset

  echo ""
  log "Stack deployed successfully!"
  echo ""

  # Print all outputs
  info "Stack Outputs:"
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' \
    --output table

  API_URL=$(get_api_url)
  echo ""
  log "Your API URL: ${CYAN}${API_URL}${NC}"
  echo ""
  warn "NAT Gateway costs ~\$0.045/hr. Run './run_all.sh destroy' when done!"
}

# =============================================================================
# COMMAND: seed
# =============================================================================
cmd_seed() {
  banner "STEP 2 — Seeding DynamoDB with 20 Books"

  info "Invoking seeder Lambda..."
  RESULT=$(aws lambda invoke \
    --function-name "$FUNC_SEEDER" \
    --payload '{"action":"seed"}' \
    --cli-binary-format raw-in-base64-out \
    --region "$AWS_REGION" \
    /tmp/seed_out.json 2>&1)

  cat /tmp/seed_out.json
  echo ""
  log "Books table seeded. Verifying record count..."

  COUNT=$(aws dynamodb scan \
    --table-name "$TABLE_BOOKS" \
    --select COUNT \
    --region "$AWS_REGION" \
    --query 'Count' \
    --output text)
  log "Records in books table: $COUNT"
}

# =============================================================================
# COMMAND: test
# =============================================================================
cmd_test() {
  banner "STEP 3 — Cold Start Experiment (curl-based)"

  API_URL=$(get_api_url)
  [[ -z "$API_URL" ]] && fail "Could not get API URL. Run deploy first."
  info "API URL: $API_URL"
  echo ""

  RESULTS_FILE="cold_start_results_$(date +%Y%m%d_%H%M%S).json"
  echo "[]" > "$RESULTS_FILE"

  # ── Test 1: Force cold starts via direct Lambda invocation ──
  echo ""
  info "=== COLD START TEST via Direct Lambda Invocation ==="
  warn "Waiting 0s between invocations to get COLD starts on first call"
  info "Tip: Wait 15+ min between test runs to guarantee cold starts"
  echo ""

  PAYLOAD='{"path":"/health","httpMethod":"GET","pathParameters":null,"queryStringParameters":null,"body":null}'

  for FUNC in "$FUNC_NON_VPC" "$FUNC_VPC" "$FUNC_PROVISIONED"; do
    echo -e "${BLUE}Testing: $FUNC${NC}"
    OUT_FILE="/tmp/invoke_${FUNC}.json"

    # Capture full log output
    LOG_B64=$(aws lambda invoke \
      --function-name "$FUNC" \
      --payload "$PAYLOAD" \
      --cli-binary-format raw-in-base64-out \
      --log-type Tail \
      --region "$AWS_REGION" \
      --query 'LogResult' \
      --output text \
      "$OUT_FILE" 2>/dev/null)

    LOG=$(echo "$LOG_B64" | base64 --decode 2>/dev/null || true)
    RESPONSE=$(cat "$OUT_FILE" 2>/dev/null)

    # Parse init duration from REPORT line
    INIT_DURATION=$(echo "$LOG" | grep -oP 'Init Duration: \K[\d.]+' || echo "0 (warm)")
    DURATION=$(echo "$LOG" | grep -oP 'Duration: \K[\d.]+' | head -1 || echo "N/A")
    MEMORY_USED=$(echo "$LOG" | grep -oP 'Max Memory Used: \K[\d]+' || echo "N/A")

    echo "  Response     : $RESPONSE"
    echo "  Duration     : ${DURATION} ms"
    echo "  Init Duration: ${INIT_DURATION} ms  ← COLD START"
    echo "  Memory Used  : ${MEMORY_USED} MB"
    echo ""
  done

  # ── Test 2: API Gateway endpoint tests ──
  echo ""
  info "=== API GATEWAY ENDPOINT TESTS ==="
  echo ""

  # Health
  info "GET /health"
  RESP=$(curl -s -w "\n%{http_code} %{time_total}s" "${API_URL}/health")
  echo "$RESP"
  echo ""

  # List books
  info "GET /books"
  RESP=$(curl -s -w "\n%{http_code} %{time_total}s" "${API_URL}/books")
  echo "$RESP" | head -c 500
  echo ""
  echo ""

  # Books by genre
  info "GET /books?genre=Technology"
  RESP=$(curl -s -w "\n%{http_code} %{time_total}s" "${API_URL}/books?genre=Technology")
  echo "$RESP" | head -c 500
  echo ""
  echo ""

  # Create a book
  info "POST /books"
  CREATE_RESP=$(curl -s -X POST "${API_URL}/books" \
    -H "Content-Type: application/json" \
    -d '{"title":"Cold Start Research Paper","author":"Your Name","genre":"Technology","publishedYear":2025,"price":0.00,"stock":1}')
  echo "$CREATE_RESP"
  BOOK_ID=$(echo "$CREATE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('bookId',''))" 2>/dev/null || echo "")
  echo ""

  # Get the created book
  if [[ -n "$BOOK_ID" ]]; then
    info "GET /books/$BOOK_ID"
    curl -s "${API_URL}/books/${BOOK_ID}"
    echo ""
    echo ""

    # Update it
    info "PUT /books/$BOOK_ID"
    curl -s -X PUT "${API_URL}/books/${BOOK_ID}" \
      -H "Content-Type: application/json" \
      -d '{"stock":999,"tags":["research","cold-start"]}'
    echo ""
    echo ""

    # Delete it
    info "DELETE /books/$BOOK_ID"
    curl -s -X DELETE "${API_URL}/books/${BOOK_ID}"
    echo ""
    echo ""
  fi

  # ── Test 3: 10 rapid calls to same function (warm invocations) ──
  echo ""
  info "=== WARM INVOCATION TIMING TEST (10 rapid calls to non-vpc) ==="
  echo ""
  for i in $(seq 1 10); do
    START=$(date +%s%N)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health")
    END=$(date +%s%N)
    MS=$(( (END - START) / 1000000 ))
    printf "  Call %2d → HTTP %s  %dms\n" "$i" "$HTTP_CODE" "$MS"
  done

  echo ""
  log "Experiment complete! Check DynamoDB metrics table for stored telemetry."
  log "CloudWatch Dashboard: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=coldstart-research-dashboard"
}

# =============================================================================
# COMMAND: load (k6)
# =============================================================================
cmd_load() {
  banner "STEP 4 — k6 Load Test"

  command -v k6 >/dev/null 2>&1 || fail "k6 not installed. Run: brew install k6  (macOS)"

  API_URL=$(get_api_url)
  [[ -z "$API_URL" ]] && fail "Could not get API URL. Run deploy first."

  SCRIPT="$(dirname "$0")/load-test.js"
  [[ -f "$SCRIPT" ]] || fail "k6 script not found: $SCRIPT"

  info "Running k6 load test against: $API_URL"
  info "This runs 3 scenarios: cold_start_burst → warm_baseline → spike"
  echo ""

  k6 run --env API_URL="$API_URL" "$SCRIPT"
}

# =============================================================================
# COMMAND: extract (export DynamoDB → CSV)
# =============================================================================
cmd_extract() {
  banner "STEP 5 — Extract Metrics Dataset"

  command -v python3 >/dev/null 2>&1 || fail "python3 not found"

  SCRIPT="$(dirname "$0")/extract_dataset.py"
  [[ -f "$SCRIPT" ]] || fail "Extractor script not found: $SCRIPT"

  info "Installing dependencies..."
  pip3 install boto3 pandas tqdm --quiet --break-system-packages 2>/dev/null || \
  pip3 install boto3 pandas tqdm --quiet 2>/dev/null || true

  info "Extracting dataset (past 7 days)..."
  python3 "$SCRIPT" \
    --region "$AWS_REGION" \
    --project "$PROJECT" \
    --days 7 \
    --output "cold_start_dataset.csv"

  log "Dataset ready: cold_start_dataset.csv"
  wc -l cold_start_dataset.csv 2>/dev/null && head -3 cold_start_dataset.csv
}

# =============================================================================
# COMMAND: train (ML/DL models)
# =============================================================================
cmd_train() {
  banner "STEP 6 — Train ML/DL Models"

  command -v python3 >/dev/null 2>&1 || fail "python3 not found"

  SCRIPT="$(dirname "$0")/train_models.py"
  [[ -f "$SCRIPT" ]] || fail "Training script not found: $SCRIPT"

  info "Installing ML dependencies..."
  pip3 install pandas numpy scikit-learn xgboost matplotlib seaborn \
    --quiet --break-system-packages 2>/dev/null || \
  pip3 install pandas numpy scikit-learn xgboost matplotlib seaborn --quiet 2>/dev/null || true

  DATASET="cold_start_dataset.csv"
  if [[ ! -f "$DATASET" ]]; then
    warn "Dataset not found. Running in synthetic-data mode."
    DATASET="synthetic"
  fi

  python3 "$SCRIPT" \
    --dataset "$DATASET" \
    --output_dir "model_outputs"

  log "Model outputs saved to ./model_outputs/"
}

# =============================================================================
# COMMAND: dashboard
# =============================================================================
cmd_dashboard() {
  banner "Opening CloudWatch Dashboard"
  URL="https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=coldstart-research-dashboard"
  info "URL: $URL"
  open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || echo "Open manually: $URL"

  echo ""
  info "Lambda Insights (per-function):"
  echo "  https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#lambda-insights:functions"

  echo ""
  info "DynamoDB Metrics Table:"
  echo "  https://us-east-1.console.aws.amazon.com/dynamodbv2/home?region=us-east-1#table?name=coldstart-research-metrics"
}

# =============================================================================
# COMMAND: status
# =============================================================================
cmd_status() {
  banner "Stack Status"
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].{Status:StackStatus,Created:CreationTime,Updated:LastUpdatedTime}' \
    --output table 2>/dev/null || warn "Stack not found or not deployed yet."

  echo ""
  info "Outputs:"
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' \
    --output table 2>/dev/null || true

  echo ""
  info "DynamoDB record counts:"
  for TABLE in "$TABLE_BOOKS" "$TABLE_METRICS"; do
    COUNT=$(aws dynamodb scan \
      --table-name "$TABLE" \
      --select COUNT \
      --region "$AWS_REGION" \
      --query 'Count' \
      --output text 2>/dev/null || echo "N/A")
    echo "  $TABLE : $COUNT records"
  done
}

# =============================================================================
# COMMAND: destroy
# =============================================================================
cmd_destroy() {
  banner "TEARDOWN — Deleting Stack (saves cost)"
  warn "This will DELETE all resources: Lambda, DynamoDB, VPC, NAT Gateway, API Gateway."
  echo ""
  read -rp "Type 'yes' to confirm deletion: " CONFIRM
  [[ "$CONFIRM" == "yes" ]] || { info "Cancelled."; exit 0; }

  info "Deleting stack: $STACK_NAME"
  aws cloudformation delete-stack \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION"

  info "Waiting for deletion to complete..."
  aws cloudformation wait stack-delete-complete \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" 2>/dev/null || true

  log "Stack deleted. NAT Gateway billing stopped."
  warn "Your CSV dataset and model outputs are still in your local folder."
}

# =============================================================================
# ROUTER
# =============================================================================
COMMAND="${1:-help}"
case "$COMMAND" in
  deploy)    cmd_deploy    ;;
  seed)      cmd_seed      ;;
  test)      cmd_test      ;;
  load)      cmd_load      ;;
  extract)   cmd_extract   ;;
  train)     cmd_train     ;;
  dashboard) cmd_dashboard ;;
  status)    cmd_status    ;;
  destroy)   cmd_destroy   ;;
  help|*)
    banner "Cold Start Research — Command Reference"
    echo "  Account : 873166938412"
    echo "  Region  : us-east-1"
    echo "  Memory  : 128 MB"
    echo ""
    echo "  Commands:"
    echo "    ./run_all.sh deploy     ← Deploy CloudFormation stack (~5-8 min)"
    echo "    ./run_all.sh seed       ← Seed DynamoDB with 20 books"
    echo "    ./run_all.sh test       ← Run cold start experiment (no extra tools)"
    echo "    ./run_all.sh load       ← Run k6 load test (install: brew install k6)"
    echo "    ./run_all.sh extract    ← Export DynamoDB + CloudWatch → CSV"
    echo "    ./run_all.sh train      ← Train ML/DL models on CSV"
    echo "    ./run_all.sh dashboard  ← Open CloudWatch dashboard in browser"
    echo "    ./run_all.sh status     ← Check stack and DB status"
    echo "    ./run_all.sh destroy    ← Delete all AWS resources (saves cost!)"
    echo ""
    warn "NAT Gateway costs ~\$0.045/hr. Run destroy when not testing."
    ;;
esac