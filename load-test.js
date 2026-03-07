/**
 * k6 Load Test Script — Cold Start Latency Research
 * 
 * Usage:
 *   k6 run --env API_URL=https://xxxx.execute-api.us-east-1.amazonaws.com/dev load-test.js
 *
 * Scenarios:
 *   1. cold_start_burst  — Burst traffic to trigger cold starts (low sleep between VUs)
 *   2. warm_baseline     — Steady traffic to measure warm execution
 *   3. spike_test        — Sudden spike to measure concurrency scaling
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Trend, Rate } from 'k6/metrics';

// ── Custom metrics ──────────────────────────────────────────
const coldStartCount     = new Counter('cold_start_count');
const warmStartCount     = new Counter('warm_start_count');
const coldStartDuration  = new Trend('cold_start_duration_ms');
const warmStartDuration  = new Trend('warm_start_duration_ms');
const errorRate          = new Rate('error_rate');

// ── Config ───────────────────────────────────────────────────
const BASE_URL = __ENV.API_URL || 'https://REPLACE_WITH_YOUR_API_URL/dev';

// All 3 function paths (via same API Gateway, different stages or paths)
// For simplicity we test the non-vpc endpoint; repeat with vpc/provisioned URLs
const ENDPOINTS = {
  nonVpc:      BASE_URL,
  // vpc:      __ENV.VPC_URL || BASE_URL,    // Uncomment after deploying stage variants
  // provisioned: __ENV.PROV_URL || BASE_URL,
};

// ── Test scenarios ────────────────────────────────────────────
export const options = {
  scenarios: {

    // Scenario 1: Force cold starts by idling then bursting
    cold_start_burst: {
      executor: 'per-vu-iterations',
      vus: 20,
      iterations: 1,
      maxDuration: '3m',
      startTime: '0s',
      tags: { scenario: 'cold_burst' },
    },

    // Scenario 2: Warm baseline — sustained moderate load
    warm_baseline: {
      executor: 'constant-arrival-rate',
      rate: 10,
      timeUnit: '1s',
      duration: '2m',
      preAllocatedVUs: 15,
      maxVUs: 30,
      startTime: '3m',
      tags: { scenario: 'warm_baseline' },
    },

    // Scenario 3: Spike — sudden burst of 50 VUs
    spike_test: {
      executor: 'ramping-vus',
      stages: [
        { duration: '10s', target: 50 },
        { duration: '30s', target: 50 },
        { duration: '10s', target: 0 },
      ],
      startTime: '6m',
      tags: { scenario: 'spike' },
    },
  },

  thresholds: {
    http_req_duration:         ['p(95)<3000'],
    http_req_failed:           ['rate<0.05'],
    cold_start_duration_ms:    ['avg<2000', 'p(95)<4000'],
    warm_start_duration_ms:    ['avg<200',  'p(95)<500'],
  },
};

// ── Helper ────────────────────────────────────────────────────
function parseHeaders(res) {
  return {
    isCold:       (res.headers['X-Cold-Start'] || 'false').toLowerCase() === 'true',
    functionType: res.headers['X-Function-Type'] || 'unknown',
    durationMs:   parseFloat(res.headers['X-Duration-Ms'] || '0'),
  };
}

// ── Main test function ────────────────────────────────────────
export default function () {
  const scenario = __ENV.SCENARIO || 'default';

  // 1. Health check
  const health = http.get(`${ENDPOINTS.nonVpc}/health`);
  check(health, { 'health 200': r => r.status === 200 });
  const hMeta = parseHeaders(health);
  if (hMeta.isCold) {
    coldStartCount.add(1);
    coldStartDuration.add(hMeta.durationMs);
  } else {
    warmStartCount.add(1);
    warmStartDuration.add(hMeta.durationMs);
  }
  errorRate.add(health.status !== 200);

  sleep(0.5);

  // 2. List books
  const list = http.get(`${ENDPOINTS.nonVpc}/books`);
  check(list, { 'list books 200': r => r.status === 200 });
  errorRate.add(list.status !== 200);

  sleep(0.3);

  // 3. Create a book
  const payload = JSON.stringify({
    title:         `Test Book ${Date.now()}`,
    author:        'k6 Test Runner',
    genre:         'Technology',
    publishedYear: 2024,
    price:         9.99,
    stock:         10,
  });

  const created = http.post(`${ENDPOINTS.nonVpc}/books`, payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  check(created, { 'create book 201': r => r.status === 201 });
  errorRate.add(created.status !== 201);

  // 4. Get the created book
  if (created.status === 201) {
    let body = {};
    try { body = JSON.parse(created.body); } catch {}
    if (body.bookId) {
      const get = http.get(`${ENDPOINTS.nonVpc}/books/${body.bookId}`);
      check(get, { 'get book 200': r => r.status === 200 });
    }
  }

  // 5. Query by genre
  const genre = http.get(`${ENDPOINTS.nonVpc}/books?genre=Technology`);
  check(genre, { 'genre filter 200': r => r.status === 200 });

  sleep(1);
}

// ── Summary handler ───────────────────────────────────────────
export function handleSummary(data) {
  const summary = {
    timestamp:            new Date().toISOString(),
    cold_start_count:     data.metrics.cold_start_count?.values?.count || 0,
    warm_start_count:     data.metrics.warm_start_count?.values?.count || 0,
    cold_start_avg_ms:    data.metrics.cold_start_duration_ms?.values?.avg?.toFixed(2) || 'N/A',
    cold_start_p95_ms:    data.metrics.cold_start_duration_ms?.values?.['p(95)']?.toFixed(2) || 'N/A',
    warm_start_avg_ms:    data.metrics.warm_start_duration_ms?.values?.avg?.toFixed(2) || 'N/A',
    warm_start_p95_ms:    data.metrics.warm_start_duration_ms?.values?.['p(95)']?.toFixed(2) || 'N/A',
    http_req_avg_ms:      data.metrics.http_req_duration?.values?.avg?.toFixed(2) || 'N/A',
    http_req_p95_ms:      data.metrics.http_req_duration?.values?.['p(95)']?.toFixed(2) || 'N/A',
    http_req_p99_ms:      data.metrics.http_req_duration?.values?.['p(99)']?.toFixed(2) || 'N/A',
    error_rate:           data.metrics.error_rate?.values?.rate?.toFixed(4) || 'N/A',
  };

  console.log('\n========== COLD START RESEARCH SUMMARY ==========');
  console.log(JSON.stringify(summary, null, 2));

  return {
    'k6-summary.json': JSON.stringify(data, null, 2),
    'cold-start-results.json': JSON.stringify(summary, null, 2),
  };
}
