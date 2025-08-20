

------------------------------------------------------ DEALS ------------------------------------------------------

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
        INNER JOIN andes.pdm.dim_promotion p
            ON f.promotion_key = p.promotion_key
        INNER JOIN andes.booker.d_mp_asin_attributes maa
            ON maa.asin = f.asin
            AND maa.marketplace_id = f.marketplace_key
            AND maa.region_id = f.region_id
        INNER JOIN andes.BOOKER.D_MP_ASIN_MANUFACTURER mam
            ON mam.asin = f.asin
            AND mam.marketplace_id = 7
            AND mam.region_id = 1 
        INNER JOIN andes.roi_ml_ddl.VENDOR_COMPANY_CODES v
            ON v.vendor_code = mam.dama_mfg_vendor_code
    WHERE p.marketplace_key = 7
        AND maa.gl_product_group = 199  --edit
        AND v.company_code = 'BO92F' --edit
        AND p.approval_status IN ('Approved', 'Scheduled')
        AND p.promotion_type IN ('Best Deal', 'Deal of the Day', 'Lightning Deal', 'Event Deal')
        AND UPPER(p.promotion_internal_title) NOT LIKE '%OIH%'
        -- AND UPPER(p.promotion_internal_title) NOT LIKE '%LEAD%'
        AND TO_DATE(p.start_datetime, 'YYYY-MM-DD') 
            BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')  - interval '730 days' -- should be 730 days
            AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') 
); 


/**********************************************************************
Classifies promotions into major event types
https://w.amazon.com/bin/view/Canada_Marketing/Events/2025_Events/
**********************************************************************/

DROP TABLE IF EXISTS promotion_details;
CREATE TEMP TABLE promotion_details AS (

    WITH event_priority AS (
        SELECT 
            asin,
            customer_shipment_item_id,
            event_name,
            event_year,
            event_month,
            promo_start_date,
            promo_end_date,
            (CASE event_name
                WHEN 'PRIME DAY' THEN 1
                WHEN 'BSS' THEN 1
                WHEN 'PBDD' THEN 1
                WHEN 'T5/11/12' THEN 1
                WHEN 'BLACK FRIDAY' THEN 2
                WHEN 'CYBER MONDAY' THEN 3
                WHEN 'BOXING WEEK' THEN 4
                WHEN 'NYNY' THEN 5
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
                    customer_shipment_item_id
                ORDER BY 
                    event_priority_order,
                    promo_start_date
            ) as event_rank
        FROM event_priority
    )

    SELECT DISTINCT
        asin,
        customer_shipment_item_id,
        event_name,
        event_year,
        event_month,
        promo_start_date,
        promo_end_date
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
            ROW_NUMBER() OVER (
                PARTITION BY 
                    event_name, event_year
                ORDER BY COUNT(*) DESC
            ) as rn
        FROM promotion_details
        WHERE event_name != 'NO_PROMOTION' 
        GROUP BY 
            event_name,
            event_year,
            event_month,
            promo_start_date,
            promo_end_date
    )
    SELECT 
        event_name,
        event_year,
        event_month,
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
        COALESCE(e.event_month, DATE_PART('month', p.promo_start_date)) as event_month,
        -- Use standard dates if they exist, otherwise use original dates
        COALESCE(e.promo_start_date, p.promo_start_date) as promo_start_date,
        COALESCE(e.promo_end_date, p.promo_end_date) as promo_end_date
    FROM promotion_details p
        LEFT JOIN event_standards e
        ON p.event_name = e.event_name
        AND DATE_PART('year', p.promo_start_date) = e.event_year
    WHERE p.event_name != 'NO_PROMOTION'
);


DROP TABLE IF EXISTS deal_base;
CREATE TEMP TABLE deal_base AS (
    SELECT DISTINCT
        customer_shipment_item_id,
        asin,
        event_name,
        event_year,
        event_month,
        promo_start_date,
        promo_end_date,
        -- Calculate event duration once
        (CASE 
            WHEN promo_end_date >= TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
                THEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - promo_start_date + 1
            ELSE promo_end_date - promo_start_date + 1
        END) AS event_duration_days
    FROM consolidated_promos
);


------------------------------------------------------ ORDERS ------------------------------------------------------

/*******#********************
Base Orders Query
- Includes only retail merchant orders with shipped units > 0 
- Excludes cancelled or fraudulent orders
- Filters for last 365 days 
***************************/

DROP TABLE IF EXISTS base_orders;
CREATE TEMP TABLE base_orders AS (

    SELECT 
        o.customer_shipment_item_id,
        o.marketplace_id,
        maa.gl_product_group,
        v.company_code,
        o.asin,
        maa.item_name,
        TO_DATE(o.order_datetime, 'YYYY-MM-DD') as order_date,
        (COALESCE(o.shipped_units,0)) as shipped_units,
        (COALESCE(cp.revenue_share_amt, 0)) as revenue_share_amt        
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
            BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '730 days'
            AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
        AND o.order_condition != 6
);


-- Get deal period orders
DROP TABLE IF EXISTS deal_period_orders;
CREATE TEMP TABLE deal_period_orders AS (
    SELECT 
        b.order_date,
        d.asin,
        b.item_name,
        d.event_name,
        d.event_duration_days,
        b.company_code,
        SUM(b.shipped_units) as shipped_units,
        SUM(b.revenue_share_amt) as revenue_share_amt,
        ROUND(
            (
                CASE 
                WHEN SUM(COALESCE(b.shipped_units, 0)) = 0 THEN NULL
                ELSE SUM(COALESCE(b.revenue_share_amt, 0)) / SUM(COALESCE(b.shipped_units, 0))
                END
            )
        , 2) as asp
    FROM deal_base d
        INNER JOIN base_orders b 
        ON d.asin = b.asin
        AND d.customer_shipment_item_id = b.customer_shipment_item_id
        AND b.order_date BETWEEN d.promo_start_date AND d.promo_end_date
    GROUP BY 
        b.order_date,
        d.asin,
        b.item_name,
        d.event_name,
        d.event_duration_days,
        b.company_code
);


DROP TABLE IF EXISTS non_deal_period_orders;
CREATE TEMP TABLE non_deal_period_orders AS (
    WITH cte AS (
        SELECT 
            b.customer_shipment_item_id,
            b.order_date,
            b.asin,
            b.item_name,
            CAST('NO DEAL' as VARCHAR) as event_name,
            CAST(0 AS INT) as event_duration_days,
            b.company_code,
            b.shipped_units as shipped_units,
            b.revenue_share_amt as revenue_share_amt
        FROM base_orders b
        WHERE b.customer_shipment_item_id not in (
            select customer_shipment_item_id
            from deal_base
        )
    )

    SELECT 
        order_date,
        asin,
        item_name,
        event_name,
        event_duration_days,
        company_code,
        SUM(shipped_units) AS shipped_units,
        SUM(revenue_share_amt) AS revenue_share_amt,
        ROUND(
            (
                CASE 
                WHEN SUM(COALESCE(shipped_units, 0)) = 0 THEN NULL
                ELSE SUM(COALESCE(revenue_share_amt, 0)) / SUM(COALESCE(shipped_units, 0))
                END
            )
    , 2) as asp
    FROM cte
    GROUP BY 
        order_date,
        asin,
        item_name,
        event_name,
        event_duration_days,
        company_code
);


---------------------------------------- ORDERS (DAILY LEVEL) ----------------------------------------
DROP TABLE IF EXISTS all_orders;
CREATE TEMP TABLE all_orders AS (
    SELECT *
    FROM deal_period_orders
    UNION ALL
    SELECT *
    FROM non_deal_period_orders
);


SELECT * FROM all_orders;