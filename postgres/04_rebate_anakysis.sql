-- ============================================================================
-- Rebate Capture Analysis
-- Mirrors python/04_rebate_analysis.py — same business questions,
-- expressed in SQL.
-- ============================================================================


 
-- ----------------------------------------------------------------------------
-- 0. Same assumption as the Python notebook: 
--    rebate contracts scale with drug tier, richest on specialty.
--    Defined once here so every query below stays consistent.
-- ----------------------------------------------------------------------------
-- target_pct_by_tier: 1 -> 0.01, 2 -> 0.05, 3 -> 0.15, 4 -> 0.22
 
 
-- ----------------------------------------------------------------------------
-- 1. Overall rebate capture rate
--    Only paid (non-denied) claims carry a rebate, a denied claim was
--    never paid for, so there's nothing for the manufacturer to rebate.
-- ----------------------------------------------------------------------------
WITH paid AS (
    SELECT
        c.*,
        d.drug_tier,
        c.awp * (CASE d.drug_tier
            WHEN 1 THEN 0.01 WHEN 2 THEN 0.05
            WHEN 3 THEN 0.15 WHEN 4 THEN 0.22 END) AS expected_rebate
    FROM claims c
    JOIN drugs d ON c.ndc_code = d.ndc_code
    WHERE NOT c.denied
)
SELECT
    COUNT(*) AS paid_claims,
    SUM(rebate_amount) AS total_actual_rebate,
    SUM(expected_rebate) AS total_expected_rebate,
    SUM(rebate_amount) / SUM(expected_rebate) AS overall_capture_rate,
    SUM(expected_rebate) - SUM(rebate_amount) AS total_gap_dollars
FROM paid;
 

 
-- ----------------------------------------------------------------------------
-- 2. Capture rate by drug tier
-- ----------------------------------------------------------------------------
WITH paid AS (
    SELECT
        c.*,
        d.drug_tier,
        c.awp * (CASE d.drug_tier
            WHEN 1 THEN 0.01 WHEN 2 THEN 0.05
            WHEN 3 THEN 0.15 WHEN 4 THEN 0.22 END) AS expected_rebate
    FROM claims c
    JOIN drugs d ON c.ndc_code = d.ndc_code
    WHERE NOT c.denied
)
SELECT
    drug_tier,
    COUNT(*) AS n_claims,
    ROUND(SUM(rebate_amount) / SUM(expected_rebate), 4) AS capture_rate
FROM paid
GROUP BY drug_tier
ORDER BY drug_tier;
 

 
-- ----------------------------------------------------------------------------
-- 3. Capture rate by client
-- ----------------------------------------------------------------------------
WITH paid AS (
    SELECT
        c.*,
        d.drug_tier,
        m.client_id,
        c.awp * (CASE d.drug_tier
            WHEN 1 THEN 0.01 WHEN 2 THEN 0.05
            WHEN 3 THEN 0.15 WHEN 4 THEN 0.22 END) AS expected_rebate
    FROM claims c
    JOIN drugs d ON c.ndc_code = d.ndc_code
    JOIN members m ON c.member_id = m.member_id
    WHERE NOT c.denied
)
SELECT
    cl.client_id,
    cl.client_name,
    COUNT(*) AS n_claims,
    ROUND(SUM(p.rebate_amount), 0) AS total_actual,
    ROUND(SUM(p.expected_rebate), 0) AS total_expected,
    ROUND(SUM(p.rebate_amount) / SUM(p.expected_rebate), 4) AS capture_rate,
    ROUND(SUM(p.expected_rebate) - SUM(p.rebate_amount), 0) AS gap_dollars
FROM paid p
JOIN clients cl ON p.client_id = cl.client_id
GROUP BY cl.client_id, cl.client_name
ORDER BY capture_rate ASC
LIMIT 10;
 
 
-- ----------------------------------------------------------------------------
-- 4. Volume caveat: does capture rate get noisier for smaller clients?
--    NTILE(3) splits clients into three equal-sized volume buckets based on
--    claim count, the SQL equivalent of pd.qcut() from the Python notebook.
-- ----------------------------------------------------------------------------
WITH paid AS (
    SELECT
        c.*,
        d.drug_tier,
        m.client_id,
        c.awp * (CASE d.drug_tier
            WHEN 1 THEN 0.01 WHEN 2 THEN 0.05
            WHEN 3 THEN 0.15 WHEN 4 THEN 0.22 END) AS expected_rebate
    FROM claims c
    JOIN drugs d ON c.ndc_code = d.ndc_code
    JOIN members m ON c.member_id = m.member_id
    WHERE NOT c.denied
),
by_client AS (
    SELECT
        client_id,
        COUNT(*) AS n_claims,
        SUM(rebate_amount) / SUM(expected_rebate) AS capture_rate
    FROM paid
    GROUP BY client_id
),
bucketed AS (
    SELECT
        *,
        NTILE(3) OVER (ORDER BY n_claims) AS size_bucket
    FROM by_client
)
SELECT
    CASE size_bucket WHEN 1 THEN 'small' WHEN 2 THEN 'medium' WHEN 3 THEN 'large' END AS size_bucket,
    ROUND(AVG(n_claims), 0) AS avg_n_claims,
    ROUND(STDDEV(capture_rate)::numeric, 4) AS capture_rate_std
FROM bucketed
GROUP BY size_bucket
ORDER BY size_bucket;


 
-- ----------------------------------------------------------------------------
-- 5. Capture rate trend over time (monthly)
-- ----------------------------------------------------------------------------
WITH paid AS (
    SELECT
        c.*,
        d.drug_tier,
        c.awp * (CASE d.drug_tier
            WHEN 1 THEN 0.01 WHEN 2 THEN 0.05
            WHEN 3 THEN 0.15 WHEN 4 THEN 0.22 END) AS expected_rebate
    FROM claims c
    JOIN drugs d ON c.ndc_code = d.ndc_code
    WHERE NOT c.denied
)
SELECT
    DATE_TRUNC('month', fill_date) AS month,
    COUNT(*) AS n_claims,
    ROUND(SUM(rebate_amount) / SUM(expected_rebate), 4) AS capture_rate
FROM paid
GROUP BY DATE_TRUNC('month', fill_date)
ORDER BY month;
 
