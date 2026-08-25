-- ============================================================================
-- Client Summary View
--
-- One row per client, rolling up the four core metrics from denial_rate.sql,
-- pa_sla_breach.sql, rebate_capture.sql, and pmpm_trend.sql into a single
-- reusable view. Built so Excel/Power BI can connect directly to this view
-- instead of re-deriving these joins and CTEs from scratch — the same
-- feature table that fed the Python segmentation notebook (06), now living
-- in the database instead of a pandas DataFrame.
--
-- Usage: SELECT * FROM client_summary;
-- ============================================================================


 
CREATE OR REPLACE VIEW client_summary AS
WITH member_counts AS (
    SELECT client_id, COUNT(*) AS n_members
    FROM members
    GROUP BY client_id
),
-- --- Denial rate (mirrors denial_rate.sql query #3) -------------------------
claim_metrics AS (
    SELECT
        m.client_id,
        COUNT(*) AS n_claims,
        AVG(CASE WHEN c.denied THEN 1.0 ELSE 0 END) AS denial_rate
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    GROUP BY m.client_id
),
-- --- SLA breach rate, using each client's OWN pa_sla_hours -------------------
pa_metrics AS (
    SELECT
        m.client_id,
        COUNT(*) AS n_pa,
        AVG(CASE
            WHEN EXTRACT(EPOCH FROM (pr.decision_ts - pr.submitted_ts)) / 3600 > cl.pa_sla_hours
            THEN 1 ELSE 0
        END) AS breach_rate
    FROM pa_requests pr
    JOIN claims c ON pr.claim_id = c.claim_id
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
    GROUP BY m.client_id
),
-- --- Rebate capture (mirrors rebate_capture.sql query #3) -------------------
rebate_metrics AS (
    SELECT
        m.client_id,
        SUM(c.rebate_amount) AS total_actual_rebate,
        SUM(c.awp * (CASE d.drug_tier
            WHEN 1 THEN 0.01 WHEN 2 THEN 0.05
            WHEN 3 THEN 0.15 WHEN 4 THEN 0.22 END)) AS total_expected_rebate
    FROM claims c
    JOIN drugs d ON c.ndc_code = d.ndc_code
    JOIN members m ON c.member_id = m.member_id
    WHERE NOT c.denied
    GROUP BY m.client_id
),
-- --- PMPM (mirrors pmpm_trend.sql query #3 baseline, whole-period average) --
pmpm_metrics AS (
    SELECT
        m.client_id,
        SUM(c.plan_paid) AS total_plan_paid
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    GROUP BY m.client_id
)
SELECT
    cl.client_id,
    cl.client_name,
    cl.plan_type,
    cl.pa_sla_hours,
    cl.is_problem_account,
    mc.n_members,
    cm.n_claims,
    cm.denial_rate,
    pm.n_pa,
    pm.breach_rate,
    rm.total_actual_rebate,
    rm.total_expected_rebate,
    rm.total_actual_rebate / NULLIF(rm.total_expected_rebate, 0) AS rebate_capture_rate,
    (rm.total_expected_rebate - rm.total_actual_rebate) AS rebate_gap_dollars,
     -- 12-month period, same denominator logic as the PMPM notebook/query
    pmm.total_plan_paid / mc.n_members / 12 AS avg_pmpm
FROM clients cl
JOIN member_counts mc ON cl.client_id = mc.client_id
LEFT JOIN claim_metrics cm ON cl.client_id = cm.client_id
LEFT JOIN pa_metrics pm ON cl.client_id = pm.client_id
LEFT JOIN rebate_metrics rm ON cl.client_id = rm.client_id
LEFT JOIN pmpm_metrics pmm ON cl.client_id = pmm.client_id;
 

 
-- ----------------------------------------------------------------------------
-- Sanity checks against the view
-- ----------------------------------------------------------------------------
 
-- Should return 50 rows, one per client, matching clients.csv
SELECT COUNT(*) AS n_clients FROM client_summary;
 
-- Same volume-noise pattern the whole project has relied on: filter to
-- clients with enough members before trusting any of these rates —
-- consistent with MIN_MEMBERS = 100 in the Python notebooks.
SELECT *
FROM client_summary
WHERE n_members >= 100
ORDER BY denial_rate DESC
LIMIT 10;
 
-- Confirm known problem accounts show up with elevated metrics across
-- the board (cross-check against Python notebooks)
SELECT client_id, client_name, is_problem_account, n_members,
       denial_rate, breach_rate, rebate_capture_rate, avg_pmpm
FROM client_summary
WHERE is_problem_account
ORDER BY client_id;