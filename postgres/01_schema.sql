-- I. PBM claims schema for Postgres

DROP TABLE IF EXISTS pa_requests;
DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS members;
DROP TABLE IF EXISTS drugs;
DROP TABLE IF EXISTS clients;

CREATE TABLE clients (
    client_id           VARCHAR(10) PRIMARY KEY,
    client_name         VARCHAR(50),
    plan_type           VARCHAR(20),
    pa_sla_hours        INT,
    is_problem_account  BOOLEAN
);

CREATE TABLE drugs (
    ndc_code       VARCHAR(20) PRIMARY KEY,
    drug_name      VARCHAR(50),
    drug_tier      INT,
    requires_pa    BOOLEAN,
    base_awp       NUMERIC(10,2)
);

CREATE TABLE members (
    member_id   VARCHAR(10) PRIMARY KEY,
    client_id   VARCHAR(10) REFERENCES clients(client_id),
    plan_tier   VARCHAR(10)
);

CREATE TABLE claims (
    claim_id         VARCHAR(10) PRIMARY KEY,
    member_id        VARCHAR(10) REFERENCES members(member_id),
    ndc_code         VARCHAR(20) REFERENCES drugs(ndc_code),
    fill_date        DATE,
    days_supply      INT,
    awp              NUMERIC(10,2),
    dispensing_fee   NUMERIC(6,2),
    copay            NUMERIC(8,2),
    plan_paid        NUMERIC(10,2),
    rebate_amount    NUMERIC(10,2),
    denied           BOOLEAN,
    denial_reason    VARCHAR(50),
    pharmacy_type    VARCHAR(20)
);

CREATE TABLE pa_requests (
    pa_id          VARCHAR(12) PRIMARY KEY,
    claim_id       VARCHAR(10) REFERENCES claims(claim_id),
    submitted_ts   TIMESTAMP,
    decision_ts    TIMESTAMP,
    urgent_flag    BOOLEAN,
    decision       VARCHAR(20)
);



-- II. Create helpful indexes for the analyses

CREATE INDEX idx_claims_member ON claims(member_id);
CREATE INDEX idx_claims_ndc ON claims(ndc_code);
CREATE INDEX idx_claims_fill_date ON claims(fill_date);
CREATE INDEX idx_members_client ON members(client_id);
CREATE INDEX idx_pa_claim ON pa_requests(claim_id);



-- III. Load the CSVs with \copy

/*
In psql, at your current database and from folder containing the CSVs, run:

\copy clients FROM 'clients.csv' WITH (FORMAT csv, HEADER true)
\copy drugs FROM 'drugs.csv' WITH (FORMAT csv, HEADER true)
\copy members FROM 'members.csv' WITH (FORMAT csv, HEADER true)
\copy claims FROM 'claims.csv' WITH (FORMAT csv, HEADER true)
\copy pa_requests FROM 'pa_requests.csv' WITH (FORMAT csv, HEADER true)

Load order matters: clients/drugs before members, members before claims,
claims before pa_requests, because of the foreign key constraints.
*/



-- IV. Sanity check

SELECT COUNT(*) FROM clients;
SELECT COUNT(*) FROM drugs;
SELECT COUNT(*) FROM members;
SELECT COUNT(*) FROM claims;
SELECT COUNT(*) FROM pa_requests;