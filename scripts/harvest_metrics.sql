-- Live-run metrics harvest. Produces the grounded numbers for the resume.
--
-- Run after the classification + resolution batches have drained:
--   docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml \
--     exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
--     < scripts/harvest_metrics.sql
--
-- Everything here reads telemetry the workers already wrote (llm_logs,
-- resolutions, complaints) — no estimation, no fabrication.

\echo '== 1. CLASSIFIER: throughput, latency, fallback (operation = classify) =='
SELECT
  count(*)                                                        AS classify_calls,
  round(avg(latency_ms))                                          AS mean_ms,
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)) AS p50_ms,
  round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)) AS p95_ms,
  round(count(*) / NULLIF(EXTRACT(EPOCH FROM (max(created_at) - min(created_at))) / 60, 0), 1)
                                                                  AS per_minute,
  round(100.0 * avg((was_fallback)::int), 2)                      AS pct_cloud_fallback
FROM llm_logs
WHERE operation = 'classify';

\echo ''
\echo '== 2. ALL OPS: per operation x provider — calls, latency, tokens, cost =='
SELECT
  operation,
  provider,
  count(*)                                                        AS calls,
  round(percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)) AS p50_ms,
  round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)) AS p95_ms,
  sum(coalesce(prompt_tokens, 0) + coalesce(completion_tokens, 0)) AS tokens,
  round(sum(coalesce(cost_usd, 0))::numeric, 4)                   AS cost_usd
FROM llm_logs
GROUP BY operation, provider
ORDER BY operation, provider;

\echo ''
\echo '== 3. GUARDRAILS: outcome distribution (resolutions.guardrail_status) =='
SELECT guardrail_status, count(*) AS n,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM resolutions
GROUP BY guardrail_status
ORDER BY n DESC;

\echo ''
\echo '== 4. GUARDRAILS: regeneration depth (version = attempts to pass) =='
SELECT version, count(*) AS n
FROM resolutions
GROUP BY version
ORDER BY version;

\echo ''
\echo '== 5. GUARDRAILS: violation breakdown by layer + code =='
SELECT v->>'layer' AS layer, v->>'code' AS code, count(*) AS n
FROM resolutions,
     LATERAL jsonb_array_elements(coalesce(guardrail_violations::jsonb, '[]'::jsonb)) AS v
GROUP BY 1, 2
ORDER BY n DESC;

\echo ''
\echo '== 6. CLASSIFICATION DISTRIBUTION: sentiment =='
SELECT sentiment, count(*) AS n
FROM complaints WHERE sentiment IS NOT NULL
GROUP BY sentiment ORDER BY n DESC;

\echo ''
\echo '== 7. CLASSIFICATION DISTRIBUTION: intent =='
SELECT intent, count(*) AS n
FROM complaints WHERE intent IS NOT NULL
GROUP BY intent ORDER BY n DESC;

\echo ''
\echo '== 8. CLASSIFICATION DISTRIBUTION: urgency (1-5) =='
SELECT urgency, count(*) AS n
FROM complaints WHERE urgency IS NOT NULL
GROUP BY urgency ORDER BY urgency;

\echo ''
\echo '== 9. COST: total spend + cost per resolution =='
SELECT
  round(sum(coalesce(cost_usd, 0))::numeric, 4)                   AS total_cost_usd,
  (SELECT count(*) FROM resolutions)                             AS resolutions,
  round((sum(coalesce(cost_usd, 0)) / NULLIF((SELECT count(*) FROM resolutions), 0))::numeric, 5)
                                                                  AS cost_per_resolution_usd
FROM llm_logs;
