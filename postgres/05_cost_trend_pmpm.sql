-- ============================================================================
-- PMPM (Per-Member-Per-Month) Cost Trend Analysis
-- Mirrors python/05_cost_trend_pmpm.py — same business questions,
-- expressed in SQL.
-- ============================================================================


 
-- ----------------------------------------------------------------------------
-- 1. Overall book-of-business PMPM trend, pool all clients
--    Total plan-paid dollars per month, divided by TOTAL covered members
-- ----------------------------------------------------------------------------
SELECT
    DATE_TRUNC('month', c.fill_date) AS month,
    SUM(c.plan_paid) / (SELECT COUNT(*) FROM members) AS pmpm
FROM claims c
GROUP BY DATE_TRUNC('month', c.fill_date)
ORDER BY month;
 

 
-- ----------------------------------------------------------------------------
-- 2. Cost breakdown by drug tier, by month
-- ----------------------------------------------------------------------------
SELECT
    DATE_TRUNC('month', c.fill_date) AS month,
    d.drug_tier,
    SUM(c.plan_paid) AS total_plan_paid
FROM claims c
JOIN drugs d ON c.ndc_code = d.ndc_code
GROUP BY DATE_TRUNC('month', c.fill_date), d.drug_tier
ORDER BY month, d.drug_tier;
 
-- Tier 4's share of total cost, in one number
WITH tier_totals AS (
    SELECT d.drug_tier, SUM(c.plan_paid) AS total_paid
    FROM claims c
    JOIN drugs d ON c.ndc_code = d.ndc_code
    GROUP BY d.drug_tier
)
SELECT
    (SELECT total_paid FROM tier_totals WHERE drug_tier = 4)
        / SUM(total_paid) AS tier4_share_of_total_cost
FROM tier_totals;
 

 
-- ----------------------------------------------------------------------------
-- 3. Client-level PMPM trend slope
--    REGR_SLOPE(y, x) is a built-in Postgres aggregate that fits a simple
--    linear regression and returns the slope — the direct SQL equivalent
--    of np.polyfit(x, y, 1)[0] in the Python notebook. No manual formula,
--    and no loop needed to do it per client — GROUP BY handles that.
-- ----------------------------------------------------------------------------
WITH member_counts AS (
    SELECT client_id, COUNT(*) AS n_members
    FROM members
    GROUP BY client_id
),
client_monthly AS (
    SELECT
        m.client_id,
        DATE_TRUNC('month', c.fill_date) AS month,
        SUM(c.plan_paid) AS monthly_paid
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    GROUP BY m.client_id, DATE_TRUNC('month', c.fill_date)
),
client_monthly_pmpm AS (
    SELECT
        cm.client_id,
        cm.month,
        cm.monthly_paid / mc.n_members AS pmpm,
        -- month_index: 0, 1, 2, ... per client, in chronological order —
        -- this is the "x" REGR_SLOPE needs, same role as np.arange(len(group))
        ROW_NUMBER() OVER (PARTITION BY cm.client_id ORDER BY cm.month) - 1 AS month_index,
        mc.n_members
    FROM client_monthly cm
    JOIN member_counts mc ON cm.client_id = mc.client_id
)
SELECT
    cl.client_id,
    cl.client_name,
    MAX(cmp.n_members) AS n_members,
    REGR_SLOPE(cmp.pmpm, cmp.month_index) AS pmpm_slope
FROM client_monthly_pmpm cmp
JOIN clients cl ON cmp.client_id = cl.client_id
GROUP BY cl.client_id, cl.client_name
ORDER BY pmpm_slope DESC
LIMIT 10;

 
 
-- ----------------------------------------------------------------------------
-- 4. Does client size explain slope volatility? Same volume-noise check as
--    every other notebook/query in this project, now on a regression slope.
-- ----------------------------------------------------------------------------
WITH member_counts AS (
    SELECT client_id, COUNT(*) AS n_members
    FROM members
    GROUP BY client_id
),
client_monthly AS (
    SELECT
        m.client_id,
        DATE_TRUNC('month', c.fill_date) AS month,
        SUM(c.plan_paid) AS monthly_paid
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    GROUP BY m.client_id, DATE_TRUNC('month', c.fill_date)
),
client_monthly_pmpm AS (
    SELECT
        cm.client_id,
        cm.monthly_paid / mc.n_members AS pmpm,
        ROW_NUMBER() OVER (PARTITION BY cm.client_id ORDER BY cm.month) - 1 AS month_index,
        mc.n_members
    FROM client_monthly cm
    JOIN member_counts mc ON cm.client_id = mc.client_id
),
slopes AS (
    SELECT
        client_id,
        MAX(n_members) AS n_members,
        REGR_SLOPE(pmpm, month_index) AS pmpm_slope
    FROM client_monthly_pmpm
    GROUP BY client_id
)
SELECT
    CORR(n_members, ABS(pmpm_slope)) AS size_vs_abs_slope_correlation
FROM slopes;

 
 
-- ----------------------------------------------------------------------------
-- 5. Reliable trend list: filter to clients with enough members that a
--    12-point slope fit isn't mostly noise (same MIN_MEMBERS = 100 floor
--    used in notebooks 05 and 06, for consistency).
-- ----------------------------------------------------------------------------
WITH member_counts AS (
    SELECT client_id, COUNT(*) AS n_members
    FROM members
    GROUP BY client_id
),
client_monthly AS (
    SELECT
        m.client_id,
        DATE_TRUNC('month', c.fill_date) AS month,
        SUM(c.plan_paid) AS monthly_paid
    FROM claims c
    JOIN members m ON c.member_id = m.member_id
    GROUP BY m.client_id, DATE_TRUNC('month', c.fill_date)
),
client_monthly_pmpm AS (
    SELECT
        cm.client_id,
        cm.monthly_paid / mc.n_members AS pmpm,
        ROW_NUMBER() OVER (PARTITION BY cm.client_id ORDER BY cm.month) - 1 AS month_index,
        mc.n_members
    FROM client_monthly cm
    JOIN member_counts mc ON cm.client_id = mc.client_id
)
SELECT
    cl.client_id,
    cl.client_name,
    MAX(cmp.n_members) AS n_members,
    REGR_SLOPE(cmp.pmpm, cmp.month_index) AS pmpm_slope
FROM client_monthly_pmpm cmp
JOIN clients cl ON cmp.client_id = cl.client_id
GROUP BY cl.client_id, cl.client_name
HAVING MAX(cmp.n_members) >= 100
ORDER BY pmpm_slope DESC
LIMIT 10;
 
 
-- ----------------------------------------------------------------------------
-- 6. Drill-down: tier breakdown for the single top reliable client
--    (parameterize client_id below to check any client from query #5)
-- ----------------------------------------------------------------------------
WITH member_counts AS (
    SELECT client_id, COUNT(*) AS n_members
    FROM members
    GROUP BY client_id
)
SELECT
    DATE_TRUNC('month', c.fill_date) AS month,
    d.drug_tier,
    SUM(c.plan_paid) / MAX(mc.n_members) AS tier_pmpm
FROM claims c
JOIN members m ON c.member_id = m.member_id
JOIN drugs d ON c.ndc_code = d.ndc_code
JOIN member_counts mc ON m.client_id = mc.client_id
WHERE m.client_id = 'CL0030'   -- swap in any client_id from query #5
GROUP BY DATE_TRUNC('month', c.fill_date), d.drug_tier
ORDER BY month, d.drug_tier;