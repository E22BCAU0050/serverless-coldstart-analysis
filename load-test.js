/**
 * k6 Load Test — Cold Start Latency Research
 * Account: 873166938412 | Region: us-east-1
 *
 * Install k6:  brew install k6
 * Run:         k6 run --env API_URL=<your-api-url> load-test.js
 *              OR use: ./run_all.sh load  (auto-injects URL)
 *
 * 3 Scenarios:
 *   1. cold_start_burst  — 20 VUs, 1 iteration each → forces cold starts
 *   2. warm_baseline     — 10 RPS for 2 min → measures warm latency
 *   3. spike_test        — ramp 0→50→0 VUs → measures concurrency scaling
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Trend, Rate } from 'k6/metrics';

// ── Custom cold start metrics ─────────────────────────────────────────────────
const coldStartDetected = new Counter('cold_start_detected');
const warmStartDetected = new Counter('warm_start_detected');
const coldStartDuration = new Trend('cold_start_duration_ms', true);
const warmStartDuration = new Trend('warm_start_duration_ms', true);
const errorRate         = new Rate('error_rate');

// ── API URL (injected from env or hardcode for direct runs) ──────────────────
const API_URL = __ENV.API_URL || 'https://REPLACE_ME.execute-api.us-east-1.amazonaws.com/dev';

const HEADERS = { 'Content-Type': 'application/json' };

// ── Test Options ──────────────────────────────────────────────────────────────
export const options = {
  scenarios: {

    // ── Scenario 1: COLD START BURST ────────────────────────────────────────
    // 20 VUs start simultaneously with no prior warm-up.
    // Each VU runs exactly 1 iteration — Lambda must spin up 20 new instances.
    // This maximises cold start observations in the CloudWatch InitDuration metric.
    cold_start_burst: {
      executor:    'per-vu-iterations',
      vus:         20,
      iterations:  1,
      maxDuration: '3m',
      startTime:   '0s',
      tags:        { scenario: 'cold_burst' },
    },

    // ── Scenario 2: WARM BASELINE ────────────────────────────────────────────
    // Steady 10 RPS for 2 minutes after burst.
    // Lambda instances are already warm → measures execution latency only.
    warm_baseline: {
      executor:       'constant-arrival-rate',
      rate:           10,
      timeUnit:       '1s',
      duration:       '2m',
      preAllocatedVUs: 15,
      maxVUs:         30,
      startTime:      '3m',
      tags:           { scenario: 'warm_baseline' },
    },

    // ── Scenario 3: SPIKE TEST ───────────────────────────────────────────────
    // Sudden burst to 50 VUs → tests concurrency scaling and second-wave cold starts.
    spike_test: {
      executor: 'ramping-vus',
      stages: [
        { duration: '10s', target: 50 },
        { duration: '30s', target: 50 },
        { duration: '10s', target:  0 },
      ],
      startTime: '6m',
      tags:      { scenario: 'spike' },
    },
  },

  // ── Thresholds for research pass/fail criteria ────────────────────────────
  thresholds: {
    http_req_duration:      ['p(95)<5000'],   // Overall p95 < 5s (generous for cold starts)
    http_req_failed:        ['rate<0.05'],    // Error rate < 5%
    cold_start_duration_ms: ['avg<3000'],     // Cold starts avg < 3s
    warm_start_duration_ms: ['avg<300', 'p(95)<600'], // Warm calls fast
  },
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function parseColdStart(res) {
  const isCold   = (res.headers['X-Cold-Start'] || '').toLowerCase() === 'true';
  const durationMs = parseFloat(res.headers['X-Duration-Ms'] || '0');
  return { isCold, durationMs };
}

function trackMetric(res) {
  const { isCold, durationMs } = parseColdStart(res);
  if (isCold) {
    coldStartDetected.add(1);
    if (durationMs > 0) coldStartDuration.add(durationMs);
  } else {
    warmStartDetected.add(1);
    if (durationMs > 0) warmStartDuration.add(durationMs);
  }
  errorRate.add(res.status >= 400 || res.status === 0);
}

// ── Main VU function ──────────────────────────────────────────────────────────
export default function () {
  let res;

  // 1. Health check (lightest possible payload — best for isolating cold start)
  res = http.get(`${API_URL}/health`, { tags: { endpoint: 'health' } });
  check(res, { 'health: status 200': r => r.status === 200 });
  trackMetric(res);
  sleep(0.3);

  // 2. List all books
  res = http.get(`${API_URL}/books`, { tags: { endpoint: 'list-books' } });
  check(res, {
    'list-books: status 200': r => r.status === 200,
    'list-books: has data':   r => { try { return JSON.parse(r.body).books !== undefined; } catch { return false; } }
  });
  errorRate.add(res.status !== 200);
  sleep(0.2);

  // 3. Filter by genre
  res = http.get(`${API_URL}/books?genre=Technology`, { tags: { endpoint: 'filter-genre' } });
  check(res, { 'filter-genre: status 200': r => r.status === 200 });
  sleep(0.2);

  // 4. Create a book
  const payload = JSON.stringify({
    title:         `k6 Test Book ${Date.now()}`,
    author:        'Load Test Runner',
    genre:         'Technology',
    publishedYear: 2025,
    price:         0.00,
    stock:         1,
    tags:          ['test', 'k6'],
  });
  res = http.post(`${API_URL}/books`, payload, {
    headers: HEADERS,
    tags:    { endpoint: 'create-book' },
  });
  check(res, { 'create-book: status 201': r => r.status === 201 });
  errorRate.add(res.status !== 201);

  // 5. Get and update the created book
  let bookId = '';
  try { bookId = JSON.parse(res.body).bookId || ''; } catch {}

  if (bookId) {
    sleep(0.1);
    res = http.get(`${API_URL}/books/${bookId}`, { tags: { endpoint: 'get-book' } });
    check(res, { 'get-book: status 200': r => r.status === 200 });

    sleep(0.1);
    res = http.put(`${API_URL}/books/${bookId}`,
      JSON.stringify({ stock: 0, notes: 'cleanup' }),
      { headers: HEADERS, tags: { endpoint: 'update-book' } }
    );
    check(res, { 'update-book: status 200': r => r.status === 200 });
  }

  sleep(1);
}

// ── Summary report ────────────────────────────────────────────────────────────
export function handleSummary(data) {
  const m = data.metrics;
  const get = (key, stat) => m[key]?.values?.[stat]?.toFixed(2) ?? 'N/A';

  const summary = {
    generated_at:          new Date().toISOString(),
    account:               '873166938412',
    region:                'us-east-1',
    total_requests:        m.http_reqs?.values?.count ?? 0,
    cold_starts_detected:  m.cold_start_detected?.values?.count ?? 0,
    warm_starts_detected:  m.warm_start_detected?.values?.count ?? 0,
    cold_start_avg_ms:     get('cold_start_duration_ms', 'avg'),
    cold_start_p95_ms:     get('cold_start_duration_ms', 'p(95)'),
    cold_start_p99_ms:     get('cold_start_duration_ms', 'p(99)'),
    warm_start_avg_ms:     get('warm_start_duration_ms', 'avg'),
    warm_start_p95_ms:     get('warm_start_duration_ms', 'p(95)'),
    http_req_avg_ms:       get('http_req_duration', 'avg'),
    http_req_p95_ms:       get('http_req_duration', 'p(95)'),
    http_req_p99_ms:       get('http_req_duration', 'p(99)'),
    error_rate_pct:        ((m.error_rate?.values?.rate ?? 0) * 100).toFixed(2) + '%',
  };

  console.log('\n\n' + '='.repeat(60));
  console.log('COLD START RESEARCH — k6 SUMMARY');
  console.log('='.repeat(60));
  Object.entries(summary).forEach(([k, v]) => console.log(`  ${k.padEnd(30)}: ${v}`));
  console.log('='.repeat(60) + '\n');

  return {
    'k6_raw_summary.json':        JSON.stringify(data, null, 2),
    'k6_coldstart_summary.json':  JSON.stringify(summary, null, 2),
  };
}
