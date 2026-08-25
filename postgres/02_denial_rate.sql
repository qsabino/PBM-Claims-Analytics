-- ============================================================================
-- Denial Rate Analysis
-- Mirrors python/02_denial_analysis.py — same business questions,
-- expressed in SQL.
-- ============================================================================


 
-- ----------------------------------------------------------------------------
-- 1. Overall denial rate
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total_claims,
    SUM(CASE WHEN denied THEN 1 ELSE 0 END) AS denied_claims,
    AVG(CASE WHEN denied THEN 1 ELSE 0 END) AS overall_denial_rate
FROM claims;

 
 
-- ----------------------------------------------------------------------------
-- 2. Denial rate by drug tier
-- ----------------------------------------------------------------------------
SELECT
    d.drug_tier,
    COUNT(*) AS n_claims,
    AVG(CASE WHEN c.denied THEN 1 ELSE 0 END) AS denial_rate
FROM claims c
JOIN drugs d ON c.ndc_code = d.ndc_code
GROUP BY d.drug_tier
ORDER BY d.drug_tier;
 

 
-- ----------------------------------------------------------------------------
-- 3. Denial rate by client,
--	  a proportion z-test to separate real
--    outliers from small-sample noise.
--
--    z = (client_rate - overall_rate) / sqrt(overall_rate * (1-overall_rate) / n)
-- 	  use ROUND() to round decimal numbers
-- ----------------------------------------------------------------------------
WITH overall AS (
    SELECT AVG(CASE WHEN denied THEN 1 ELSE 0 END) AS overall_rate
    FROM claims
),
client_denials AS (
    SELECT
        cl.client_id,
        cl.client_name,
        COUNT(*) AS n_claims,
        AVG(CASE WHEN c.denied THEN 1 ELSE 0 END) AS denial_rate
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
    GROUP BY cl.client_id, cl.client_name
)
SELECT
    cd.client_id,
    cd.client_name,
    cd.n_claims,
    cd.denial_rate,
    (cd.denial_rate - o.overall_rate)
        / SQRT(o.overall_rate * (1 - o.overall_rate) / cd.n_claims)
    AS z_score
FROM client_denials cd
CROSS JOIN overall o
ORDER BY z_score DESC;

 
 
-- ----------------------------------------------------------------------------
-- 4. Same as #3, but only the statistically significant outliers (z > 2)
--    This is the SQL equivalent of the `flagged` DataFrame in Python.
-- ----------------------------------------------------------------------------
WITH overall AS (
    SELECT AVG(CASE WHEN denied THEN 1 ELSE 0 END) AS overall_rate
    FROM claims
),
client_denials AS (
    SELECT
        cl.client_id,
        cl.client_name,
        COUNT(*) AS n_claims,
        AVG(CASE WHEN c.denied THEN 1 ELSE 0 END) AS denial_rate
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    JOIN clients cl ON m.client_id = cl.client_id
    GROUP BY cl.client_id, cl.client_name, cl.is_problem_account
),
scored AS (
    SELECT
        cd.*,
        (cd.denial_rate - o.overall_rate)
        / SQRT(o.overall_rate * (1 - o.overall_rate) / cd.n_claims) AS z_score
    FROM client_denials cd
    CROSS JOIN overall o
)
SELECT
    client_id, client_name, n_claims,
    ROUND(denial_rate, 4) AS denial_rate,
    ROUND(z_score, 2) AS z_score
FROM scored
WHERE z_score > 2
ORDER BY z_score DESC;
 

 
-- ----------------------------------------------------------------------------
-- 5. Denial reason breakdown — by count and by dollar (AWP) exposure
-- ----------------------------------------------------------------------------
SELECT
    denial_reason,
    COUNT(*) AS n_denials,
    100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct_of_denials,
    SUM(awp) AS awp_exposure
FROM claims
WHERE denied
GROUP BY denial_reason
ORDER BY n_denials DESC;
 

 
-- ----------------------------------------------------------------------------
-- 6. Denial rate trend over time (monthly)
-- ----------------------------------------------------------------------------
SELECT
    DATE_TRUNC('month', fill_date) AS month,
    COUNT(*) AS n_claims,
    AVG(CASE WHEN denied THEN 1 ELSE 0 END) AS denial_rate
FROM claims
GROUP BY DATE_TRUNC('month', fill_date)
ORDER BY month;
 
