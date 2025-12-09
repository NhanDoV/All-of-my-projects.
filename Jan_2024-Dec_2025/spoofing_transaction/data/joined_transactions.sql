CREATE OR REPLACE VIEW analytics.joined_transactions AS
SELECT
    -- Core transaction identifiers and timing
    t.transid,          -- Unique transaction ID
    t.transnb,          -- Transaction number from POS
    t.transdate,        -- Transaction date/time at POS
    t.total_amount,     -- Total transaction amount (can be negative for returns)
    t.dvrstart,         -- DVR timestamp (video start time)
    t.channel,          -- POS channel / lane number
    t.start_shift,      -- Shift start timestamp
    t.end_shift,        -- Shift end timestamp

    -- Employee information (from transaction and master table)
    t.employeeid,       -- Employee ID captured on transaction
    e.employee_name,    -- Employee name from employees table
    e.site_id,          -- Home site ID of the employee

    -- Site information
    s.site_name,        -- Site name from sites table
    s.city,             -- City of the site
    s.state,            -- State/Region of the site

    -- POS / customer information
    t.register_name,    -- Register / till identifier
    t.loyalty_card,     -- Loyalty card number (if captured)

    -- Raw site fields from transaction (for data quality checks)
    t.sitename          AS raw_sitename,      -- Raw site name as written on transaction
    t.siteemployee      AS raw_siteemployee,  -- Raw employee site text on transaction

    -- Exception / audit information
    t.exception_amount, -- Exception amount (if any)
    t.t_operatorid,     -- Operator ID used for exception/override
    o.operator_name,    -- Operator name from operators table
    t.t_pacid,          -- PAC ID (policy/authorization code)
    p.pac_name,         -- PAC description/name
    p.regionname,       -- Region name from PAC master
    p.division,         -- Division from PAC master
    t.exception_type,   -- Exception type text/flag

    -- Payment information
    t.payment,          -- Payment method (CASH, CreditDebit, etc.)
    t.cardno,           -- Masked card number (if applicable)

    -- Additional descriptive fields
    t."desc",           -- Free-text description / notes
    t.link,             -- Link to additional evidence (e.g. DVR, case tool)
    t.label,            -- Label / tag for the transaction
    t.has_redeem_later  -- Flag: has redeem later or not
FROM
    public.transactions t
    -- Join to employee master data
    LEFT JOIN public.employees  e ON e.employee_id  = t.employeeid
    -- Join to operator master data (for exceptions / overrides)
    LEFT JOIN public.operators  o ON o.operator_id  = t.t_operatorid
    -- Join to PAC master data (authorization / policy info)
    LEFT JOIN public.pac        p ON p.pac_id       = t.t_pacid
    -- Join to site master data (location hierarchy)
    LEFT JOIN public.sites      s ON s.site_name    = t.sitename;

SELECT *
FROM analytics.joined_transactions
WHERE transdate >= %(start_dt)s
  AND transdate <  %(end_dt)s;