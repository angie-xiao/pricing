
/*******#********************
Base Orders Query
- Includes only retail merchant orders with shipped units > 0 
- Excludes cancelled or fraudulent orders
- Filters for last 365 days 
***************************/
DROP TABLE IF EXISTS base_orders;
CREATE TEMP TABLE base_orders AS (
    SELECT 
        o.asin,
        maa.item_name,
        -- o.customer_id,
        o.customer_shipment_item_id,
        TO_DATE(o.order_datetime, 'YYYY-MM-DD') as order_date,
        o.our_price, -- price shown on amazon
        maa.brand_name,
        maa.brand_code,
        o.shipped_units,
        COALESCE(cp.revenue_share_amt, 0) as revenue_share_amt,
        maa.gl_product_group,
        mam.dama_mfg_vendor_code as vendor_code,
        v.company_code,
        v.company_name
    FROM andes.booker.d_unified_cust_shipment_items o
        INNER JOIN andes.booker.d_mp_asin_attributes maa
            ON maa.asin = o.asin
            AND maa.marketplace_id = o.marketplace_id
            AND maa.region_id = o.region_id
            AND maa.gl_product_group = 199
        LEFT JOIN andes.contribution_ddl.o_wbr_cp_na cp
            ON o.customer_shipment_item_id = cp.customer_shipment_item_id 
            AND o.asin = cp.asin
            AND cp.marketplace_id = 7
        LEFT JOIN andes.BOOKER.D_MP_ASIN_MANUFACTURER mam
            ON mam.asin = o.asin
            AND mam.marketplace_id = 7
            AND mam.region_id = 1
        LEFT JOIN andes.roi_ml_ddl.VENDOR_COMPANY_CODES v
            ON v.vendor_code = mam.dama_mfg_vendor_code
    WHERE o.region_id = 1
        AND o.marketplace_id = 7
        AND v.company_code = 'BO92F' -- boxiecat
        AND o.shipped_units > 0
        AND o.is_retail_merchant = 'Y'
        AND o.order_datetime 
            BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '60 days'
            AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
        AND o.order_condition != 6
);


/*************************
Create Deal Base with Event Dates and Handle Overlaps;
Classifies promotions into major event types
https://w.amazon.com/bin/view/Canada_Marketing/Events/2025_Events/
*************************/
DROP TABLE IF EXISTS raw_events;
CREATE TEMP TABLE raw_events AS  (
    SELECT DISTINCT
        f.asin,
        f.customer_shipment_item_id,
        TO_DATE(p.start_datetime, 'YYYY-MM-DD') as promo_start_date,
        TO_DATE(p.end_datetime, 'YYYY-MM-DD') as promo_end_date,
        DATE_PART('year', p.start_datetime) as event_year,
        DATE_PART('month', p.start_datetime) as event_month,
        (CASE 
            WHEN p.promotion_key IS NULL THEN 'NO_PROMOTION'

            -- tier 1
            WHEN UPPER(p.promotion_internal_title) LIKE '%BSS%' 
                OR UPPER(p.promotion_internal_title) LIKE '%BIG SPRING SALE%' 
                THEN 'BSS'

            -- Prime Day logic with month boundary consideration
            WHEN (DATE_PART('month', p.start_datetime) = 7 
                OR (DATE_PART('month', p.start_datetime) = 6 
                    AND DATE_PART('day', p.start_datetime) >= 25))  -- Added buffer for late June starts
                AND (
                    UPPER(p.promotion_internal_title) LIKE '%PRIME%DAY%'
                    OR UPPER(p.promotion_internal_title) LIKE '%PD%' 
                    OR UPPER(p.promotion_internal_title) LIKE '%PEBD%' 
                )
                THEN 'PRIME DAY'

            WHEN UPPER(p.promotion_internal_title) LIKE '%PBDD%' THEN 'PBDD'
            WHEN UPPER(p.promotion_internal_title) LIKE '%BF%'
                OR UPPER(p.promotion_internal_title) LIKE '%BLACK%FRIDAY%' THEN 'BLACK FRIDAY'
            WHEN UPPER(p.promotion_internal_title) LIKE '%CYBER%MONDAY%'
                OR UPPER(p.promotion_internal_title) LIKE '%CM%' THEN 'CYBER MONDAY'
            WHEN UPPER(p.promotion_internal_title) LIKE '%BOXING WEEK%'
                OR UPPER(p.promotion_internal_title) LIKE '%BOXING DAY%' THEN 'BOXING WEEK'
            WHEN UPPER(p.promotion_internal_title) LIKE '%T5%'
                OR UPPER(p.promotion_internal_title) LIKE '%T11%'
                OR UPPER(p.promotion_internal_title) LIKE '%T12%' THEN 'T5/11/12'

            -- tier 1.5
            WHEN UPPER(p.promotion_internal_title) LIKE '%BACK%TO%SCHOOL%' THEN 'BACK TO SCHOOL' 
            WHEN UPPER(p.promotion_internal_title) LIKE '%BACK%TO%UNIVERSITY%' THEN 'BACK TO UNIVERSITY' 

            -- tier 2
            WHEN UPPER(p.promotion_internal_title) LIKE '%NYNY%' THEN 'NYNY'
            -- Mother's Day with buffer
            WHEN (DATE_PART('month', p.start_datetime) = 5 
                OR (DATE_PART('month', p.start_datetime) = 4 
                    AND DATE_PART('day', p.start_datetime) >= 25))
                AND (UPPER(p.promotion_internal_title) LIKE '%MOTHER%DAY%' 
                    OR UPPER(p.promotion_internal_title) LIKE '%MOTHERS%DAY%' 
                    OR UPPER(p.promotion_internal_title) LIKE '%MOTHER_S%DAY%'
                    OR UPPER(p.promotion_internal_title) LIKE '%MOTHER''''S%DAY%')
                THEN 'MOTHERS DAY'
            WHEN UPPER(p.promotion_internal_title) LIKE '%FATHER%DAY%' 
                OR UPPER(p.promotion_internal_title) LIKE '%FATHERS%DAY%' 
                OR UPPER(p.promotion_internal_title) LIKE '%FATHER_S%DAY%'
                OR UPPER(p.promotion_internal_title) LIKE '%FATHER''''S%DAY%' THEN 'FATHERS DAY'
            WHEN UPPER(p.promotion_internal_title) LIKE '%VALENTINE%DAY%' 
                OR UPPER(p.promotion_internal_title) LIKE '%VALENTINES%DAY%'
                OR UPPER(p.promotion_internal_title) LIKE '%VALENTINE_S%DAY%'
                OR UPPER(p.promotion_internal_title) LIKE '%VALENTINE''''S%DAY%' THEN 'VALENTINES DAY'
            WHEN UPPER(p.promotion_internal_title) LIKE '%GIFTMANIA%' 
                OR UPPER(p.promotion_internal_title) LIKE '%GIFT%MANIA%' THEN 'GIFT MANIA'
            WHEN UPPER(p.promotion_internal_title) LIKE '%HALLOWEEN%' THEN 'HALLOWEEN'
            WHEN UPPER(p.promotion_internal_title) LIKE '%HOLIDAY%' THEN 'HOLIDAY'
            
            -- tier 3    
            WHEN UPPER(p.promotion_internal_title) LIKE '%LUNAR%NEW%YEAR%' THEN 'LUNAR NEW YEAR'
            WHEN UPPER(p.promotion_internal_title) LIKE '%DAILY%ESSENTIALS%' THEN 'DAILY ESSENTIALS'
            WHEN UPPER(p.promotion_internal_title) LIKE '%BEAUTY%HAUL%' THEN 'BEAUTY HAUL'
            -- Special handling for Pet Month/Day
            WHEN (DATE_PART('month', p.start_datetime) = 5 
                OR (DATE_PART('month', p.start_datetime) = 4 
                    AND DATE_PART('day', p.start_datetime) >= 25))  -- Consider late April starts as May
                AND (
                    UPPER(p.promotion_internal_title) LIKE '%PET%DAY%' 
                    OR UPPER(p.promotion_internal_title) LIKE '%PET%MONTH%'
                )
                THEN 'PET DAY'
            WHEN UPPER(p.promotion_internal_title) LIKE '%HEALTH%WELLNESS%' THEN 'HEALTH & WELLNESS MONTH'
            WHEN UPPER(p.promotion_internal_title) LIKE '%GAMING%MONTH%' THEN 'GAMING MONTH'
            WHEN UPPER(p.promotion_internal_title) LIKE '%BABY%SAVINGS%' THEN 'BABY SAVINGS'
            WHEN UPPER(p.promotion_internal_title) LIKE '%BEAUTY%WEEK%' THEN 'BEAUTY WEEK'
            WHEN UPPER(p.promotion_internal_title) LIKE '%DIWALI%' THEN 'DIWALI'
            WHEN UPPER(p.promotion_internal_title) LIKE '%MOVEMBER%' THEN 'MOVEMBER'
            WHEN UPPER(p.promotion_internal_title) LIKE '%FLASH SALE%' THEN 'FLASH SALE'
            ELSE 'OTHER'
        END) as event_name
    FROM andes.pdm.fact_promotion_cp f
        JOIN andes.pdm.dim_promotion p
        ON f.promotion_key = p.promotion_key
    WHERE p.marketplace_key = 7
        AND p.approval_status IN ('Approved', 'Scheduled')
        AND p.promotion_type IN ('Best Deal', 'Deal of the Day', 'Lightning Deal', 'Event Deal')
        AND UPPER(p.promotion_internal_title) NOT LIKE '%OIH%'
        AND UPPER(p.promotion_internal_title) NOT LIKE '%LEAD%'
        AND TO_DATE(p.start_datetime, 'YYYY-MM-DD') 
            BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '60 days'
            AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
); 


DROP TABLE IF EXISTS promotion_details;
CREATE TEMP TABLE promotion_details AS (

    WITH event_priority AS (
        SELECT 
            asin,
            customer_shipment_item_id,
            event_name,
            event_year,
            event_month,
            (CASE event_name
                WHEN 'PRIME DAY' THEN 1
                WHEN 'BLACK FRIDAY' THEN 2
                WHEN 'CYBER MONDAY' THEN 3
                WHEN 'BOXING WEEK' THEN 4
                WHEN 'BSS' THEN 5
                WHEN 'NYNY' THEN 6
                ELSE 99
            END) as event_priority_order
        FROM raw_events
    ),

    -- Handle overlaps by prioritizing events
    prioritized_events AS (
        SELECT 
            asin,
            customer_shipment_item_id,
            event_name,
            event_year,
            event_month,
            promo_start_date,
            promo_end_date,
            ROW_NUMBER() OVER (
                PARTITION BY 
                    asin,
                    event_month
                ORDER BY 
                    event_priority_order,
                    promo_start_date
            ) as event_rank
        FROM raw_events
    )

    SELECT DISTINCT
        asin,
        event_name,
        event_year,
        event_month,
        promo_start_date
    FROM prioritized_events
    WHERE event_rank = 1  -- Only take highest priority event when overlapping
);


-- Find the most common start/end dates for each event
DROP TABLE IF EXISTS event_standards;
CREATE TEMP TABLE event_standards AS (
    WITH event_counts AS (
        SELECT 
            event_name,
            event_year,
            event_month,
            promo_start_date,
            promo_end_date,
            COUNT(*) as frequency,
            -- Rank by frequency - removed month from partition
            ROW_NUMBER() OVER (
                PARTITION BY event_name, 
                DATE_PART('year', promo_start_date)
                ORDER BY COUNT(*) DESC
            ) as rn
        FROM promotion_details
        WHERE event_name != 'NO_PROMOTION'
        GROUP BY 
            event_name,
            DATE_PART('year', promo_start_date),
            promo_start_date,
            promo_end_date
    )
    SELECT 
        event_name,
        event_year,
        DATE_PART('month', promo_start_date) as event_month,  -- derived from the most common start date
        promo_start_date,
        promo_end_date,
        frequency
    FROM event_counts
    WHERE rn = 1
        AND frequency >= 3  -- Only keep patterns used by at least 3 promotions
);


-- Final consolidated promotions
DROP TABLE IF EXISTS consolidated_promos;
CREATE TEMP TABLE consolidated_promos AS (
    SELECT 
        p.asin,
        p.customer_shipment_item_id,
        p.event_name,
        DATE_PART('year', p.promo_start_date) as event_year,
        -- Use the standard event month for consistency
        COALESCE(e.event_month, DATE_PART('month', p.promo_start_date)) as event_month
    FROM promotion_details p
        LEFT JOIN event_standards e
        ON p.event_name = e.event_name
        AND DATE_PART('year', p.promo_start_date) = e.event_year
    WHERE p.event_name != 'NO_PROMOTION'
);


/*************************
Deal Metrics - Base table with date ranges
*************************/
DROP TABLE IF EXISTS deal_base;
CREATE TEMP TABLE deal_base AS (
    SELECT DISTINCT
        asin,
        customer_shipment_item_id,
        event_name,
        event_year   
    FROM consolidated_promos
);


DROP TABLE IF EXISTS unified_deal_base;
CREATE TEMP TABLE unified_deal_base AS (
    SELECT 
        -- Deal context
        d.asin,
        d.event_name,
        d.event_year,
        b.our_price,
        b.order_date,

        -- Product/Business hierarchy
        b.item_name,
        b.gl_product_group,
        b.brand_code,
        b.brand_name,
        -- b.vendor_code,
        b.company_code,
        b.company_name,

        -- Metrics
        -- b.customer_shipment_item_id,
        SUM(b.shipped_units) AS shipped_units,
        SUM(b.revenue_share_amt) AS revenue_share_amt

    FROM deal_base d
        RIGHT JOIN base_orders b 
        ON d.asin = b.asin
        AND d.customer_shipment_item_id = b.customer_shipment_item_id

    GROUP BY
        d.asin,
        d.event_name,
        d.event_year,
        -- DEAL PRICE
        b.our_price,
        -- Order date
        b.order_date,        
        -- Product/Business hierarchy
        b.item_name,
        b.gl_product_group,
        b.brand_code,
        b.brand_name,
        -- b.vendor_code,
        b.company_code,
        b.company_name
);

