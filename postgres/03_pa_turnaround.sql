-- ============================================================================
-- PA Turnaround & SLA Breach Analysis
-- Mirrors python/03_pa_turnaround.py — same business questions,
-- expressed in SQL.
-- ============================================================================


 
-- ----------------------------------------------------------------------------
-- 1. Turnaround distribution: mean vs. median vs. tail
--    EXTRACT(EPOCH FROM ...) gives the interval in seconds; /3600 converts
--    to hours, same as .dt.total_seconds()/3600 in pandas.
-- ----------------------------------------------------------------------------
WITH pa_turnaround AS (
	SELECT EXTRACT(EPOCH FROM (decision_ts - submitted_ts)) / 3600 AS turnaround_hours
	FROM pa_requests
)
SELECT
    AVG(turnaround_hours) AS mean_hours,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY turnaround_hours) AS median_hours,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY turnaround_hours) AS p90_hours,
    MIN(turnaround_hours) AS min_hours,
    MAX(turnaround_hours) AS max_hours
FROM pa_turnaround;
 

 
-- ----------------------------------------------------------------------------
-- 2. Urgent vs. standard requests (mean and median turnaround_hours)
-- ----------------------------------------------------------------------------
SELECT
    urgent_flag,
    COUNT(*) AS n_requests,
    AVG(EXTRACT(EPOCH FROM (decision_ts - submitted_ts)) / 3600) AS mean_hours,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (decision_ts - submitted_ts)) / 3600
    ) AS median_hours
FROM pa_requests
GROUP BY urgent_flag
ORDER BY urgent_flag DESC;


 
-- ----------------------------------------------------------------------------
-- 3. RAW SLA breach rate — pooled across the whole book
--    same as the Python notebook, show why
--    it's misleading before the corrected version below.
-- ----------------------------------------------------------------------------
WITH pa_client AS (
    SELECT
        pr.pa_id,
        cl.client_id,
        cl.client_name,
        cl.pa_sla_hours,
        EXTRACT(EPOCH FROM (pr.decision_ts - pr.submitted_ts)) / 3600 AS turnaround_hours
    FROM pa_requests pr
    JOIN claims c ON pr.claim_id = c.claim_id
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
)
SELECT
    client_id,
    client_name,
    pa_sla_hours,
    COUNT(*) AS n_pa,
    AVG(CASE WHEN turnaround_hours > pa_sla_hours THEN 1.0 ELSE 0 END) AS breach_rate
FROM pa_client
GROUP BY client_id, client_name, pa_sla_hours
ORDER BY breach_rate DESC
LIMIT 10;
 
 
-- ----------------------------------------------------------------------------
-- 4. RAW ranking is misleading: 
--    breach rate by SLA tier, pooled across all clients on that tier.
-- ----------------------------------------------------------------------------
WITH pa_client AS (
    SELECT
        cl.pa_sla_hours,
        EXTRACT(EPOCH FROM (pr.decision_ts - pr.submitted_ts)) / 3600 AS turnaround_hours
    FROM pa_requests pr
    JOIN claims c ON pr.claim_id = c.claim_id
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
)
SELECT
    pa_sla_hours,
    COUNT(*) AS n_pa,
    AVG(CASE WHEN turnaround_hours > pa_sla_hours THEN 1.0 ELSE 0 END) AS breach_rate
FROM pa_client
GROUP BY pa_sla_hours
ORDER BY pa_sla_hours;
 
 
-- ----------------------------------------------------------------------------
-- 5. SLA breach — the CORRECT way: z-test computed WITHIN each SLA tier.
-- ----------------------------------------------------------------------------
WITH pa_row AS (
   SELECT
        pr.pa_id,
        cl.client_id,
        cl.client_name,
        cl.is_problem_account,
        cl.pa_sla_hours,
        CASE WHEN EXTRACT(EPOCH FROM (pr.decision_ts - pr.submitted_ts)) / 3600 > cl.pa_sla_hours
             THEN 1 ELSE 0 END AS breach
    FROM pa_requests pr
    JOIN claims c ON pr.claim_id = c.claim_id
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
),
tier_rates AS (
    SELECT pa_sla_hours, AVG(breach) AS tier_rate
    FROM pa_row
    GROUP BY pa_sla_hours
),
client_breach AS (
    SELECT
        client_id,
        client_name,
        is_problem_account,
        pa_sla_hours,
        COUNT(*) AS n_pa,
        AVG(breach) AS breach_rate
    FROM pa_row
    GROUP BY client_id, client_name, is_problem_account, pa_sla_hours
)
SELECT
    cb.client_id,
    cb.client_name,
    cb.is_problem_account,
    cb.pa_sla_hours AS sla_tier,
    cb.n_pa,
    cb.breach_rate,
    (cb.breach_rate - tr.tier_rate) / SQRT(tr.tier_rate * (1 - tr.tier_rate) / cb.n_pa) AS z_score
FROM client_breach cb
JOIN tier_rates tr ON cb.pa_sla_hours = tr.pa_sla_hours
WHERE (cb.breach_rate - tr.tier_rate) / SQRT(tr.tier_rate * (1 - tr.tier_rate) / cb.n_pa) > 2
ORDER BY z_score DESC;
 

 
-- ----------------------------------------------------------------------------
-- 6. Denial rate vs. breach rate correlation, by client
--    CORR() is a built-in aggregate in Postgres
-- ----------------------------------------------------------------------------
WITH client_denial AS (
    SELECT m.client_id, AVG(CASE WHEN c.denied THEN 1.0 ELSE 0 END) AS denial_rate
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    GROUP BY m.client_id
),
client_breach AS (
    SELECT
        cl.client_id,
        AVG(CASE WHEN EXTRACT(EPOCH FROM (pr.decision_ts - pr.submitted_ts)) / 3600 > cl.pa_sla_hours
                 THEN 1 ELSE 0 END) AS breach_rate
    FROM pa_requests pr
    JOIN claims c ON pr.claim_id = c.claim_id
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
    GROUP BY cl.client_id
)
SELECT
    CORR(d.denial_rate, b.breach_rate) AS denial_breach_correlation
FROM client_denial d
JOIN client_breach b ON d.client_id = b.client_id;
 

 
-- ----------------------------------------------------------------------------
-- 7. Breach rate trend over time (by month, based on PA submission date)
-- ----------------------------------------------------------------------------
WITH pa_client AS (
    SELECT
        pr.submitted_ts,
        CASE WHEN EXTRACT(EPOCH FROM (pr.decision_ts - pr.submitted_ts)) / 3600 > cl.pa_sla_hours
             THEN 1 ELSE 0 END AS breach
    FROM pa_requests pr
    JOIN claims c ON pr.claim_id = c.claim_id
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
)
SELECT
    DATE_TRUNC('month', submitted_ts) AS month,
    COUNT(*) AS n_pa,
    ROUND(AVG(breach), 4) AS breach_rate
FROM pa_client
GROUP BY DATE_TRUNC('month', submitted_ts)
ORDER BY month;
 
