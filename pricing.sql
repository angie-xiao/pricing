/*-----------------------------------------------------------
                           shipments 
-------------------------------------------------------------*/
/* Step 1: Get filtered shipments (base layer) */
DROP TABLE IF EXISTS filtered_shipments;
CREATE TEMP TABLE filtered_shipments AS (
    SELECT 
        o.asin,
        o.customer_shipment_item_id,
        o.marketplace_id,
        o.region_id,
        round(o.our_price,2) as price,
        TO_DATE(o.order_datetime, 'YYYY-MM-DD') as order_date,
        o.shipped_units
    FROM andes.booker.d_unified_cust_shipment_items o
    WHERE o.region_id = 1
        AND o.marketplace_id = 7
        AND o.gl_product_group IN (199)
        AND o.order_datetime >= TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '730 days' -- adjustable filter 
        AND o.order_datetime <= TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
        AND o.order_condition != 6
        AND o.shipped_units > 0
        and o.our_price > 0
        AND o.is_retail_merchant = 'Y'
        AND o.asin in (
            'B06ZZ4679J','B089LLHMQL','B08NTPM4Z1','B08NTQ7XVB','B08NTSCMZS','B09P323G3B',
            'B09P32SSRL','B09P3L3BJQ','B09V65VY5J','B0B3S7JJ7G','B0B3S91RJC','B0DP36S67M'
        ) -- adjustable filter 
);


/* Step 2a: Add ASIN attributes */
DROP TABLE IF EXISTS orders_with_asin_attrs;
CREATE TEMP TABLE orders_with_asin_attrs AS (
    SELECT
        fs.asin,
        fs.customer_shipment_item_id,
        fs.marketplace_id,
        fs.region_id,
        fs.order_date,
        fs.shipped_units,
        fs.price,
        maa.item_name,
        maa.gl_product_group,
        maa.brand_name,
        maa.brand_code
    FROM filtered_shipments fs
        LEFT JOIN andes.booker.d_mp_asin_attributes maa
        ON maa.marketplace_id = fs.marketplace_id
        AND maa.region_id = fs.region_id
        AND maa.asin = fs.asin
    -- WHERE fs.asin = 'B07DK2BQGD'  -- Debugging for specific ASIN
);


/* Step 2b: Add manufacturer info */
DROP TABLE IF EXISTS orders_with_manufacturer;
CREATE TEMP TABLE orders_with_manufacturer AS (
    SELECT 
        o.*,
        mam.dama_mfg_vendor_code as vendor_code
    FROM orders_with_asin_attrs o
        LEFT JOIN andes.BOOKER.D_MP_ASIN_MANUFACTURER mam
        ON mam.marketplace_id = o.marketplace_id
        AND mam.region_id = o.region_id
        AND mam.asin = o.asin
);


/* Step 2c: Add vendor company info */
DROP TABLE IF EXISTS orders_with_vendor;
CREATE TEMP TABLE orders_with_vendor AS (
    SELECT 
        o.*,
        v.company_code,
        v.company_name
    FROM orders_with_manufacturer o
        LEFT JOIN andes.roi_ml_ddl.VENDOR_COMPANY_CODES v
        ON v.vendor_code = o.vendor_code
    WHERE v.company_code = 'CI08L' -- adjustable filter 
);

/* Step 2d: Add revenue share amount (final base_orders) */
DROP TABLE IF EXISTS base_orders;
CREATE TEMP TABLE base_orders AS (
    SELECT 
        o.asin,
        o.item_name,
        o.customer_shipment_item_id,
        o.order_date,
        o.gl_product_group,
        o.brand_name,
        o.brand_code,
        o.vendor_code,
        o.company_code,
        o.company_name,
        o.shipped_units,
        o.price,
        COALESCE(cp.revenue_share_amt, 0) as revenue_share_amt
    FROM orders_with_vendor o
        LEFT JOIN andes.contribution_ddl.o_wbr_cp_na cp
        ON o.marketplace_id = cp.marketplace_id
        AND o.customer_shipment_item_id = cp.customer_shipment_item_id
);


/*-----------------------------------------------------------
                           promos 
-------------------------------------------------------------*/
/* Step 1: Get base promotion information */
-- currently filtering for CA Pets
 DROP TABLE IF EXISTS base_promos;
CREATE TEMP TABLE base_promos AS (
    SELECT DISTINCT
        p.region_id,
        p.marketplace_key,
        CAST(pa.product_group_key AS INT) AS product_group_key,
        pa.asin,
        p.promotion_key,
        p.promotion_internal_title,
        p.start_datetime,
        p.end_datetime,
        pa.promotion_pricing_amount,
        pa.current_discount_percent
    FROM andes.pdm.dim_promotion p
        INNER JOIN andes.pdm.dim_promotion_asin pa
        ON p.promotion_key = pa.promotion_key
        AND p.region_id = pa.region_id
        AND p.marketplace_key = pa.marketplace_key
    WHERE p.region_id = 1
        AND p.marketplace_key = 7
        AND TO_DATE(start_datetime, 'YYYY-MM-DD') >= TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '730 days'   -- Changed from <= to >=
        AND TO_DATE(start_datetime, 'YYYY-MM-DD') <= TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')                         -- Added end date boundary
        AND p.approval_status IN ('Approved', 'Scheduled')
        AND p.promotion_type IN ('Best Deal', 'Deal of the Day', 'Lightning Deal', 'Event Deal')
        AND UPPER(p.promotion_internal_title) NOT LIKE '%OIH%'
        AND CAST(pa.product_group_key AS INT) =199
        -- AND pa.asin='B07DK2BQGD'  -- Debugging for specific ASIN
);


/* Step 2: Create raw events with detailed categorization */
DROP TABLE IF EXISTS raw_events;
CREATE TEMP TABLE raw_events AS (
    SELECT DISTINCT
        bp.region_id,
        bp.marketplace_key,
        bp.product_group_key,
        bp.asin,
        bp.promotion_key,
        TO_DATE(bp.start_datetime, 'YYYY-MM-DD') as promo_start_date,
        TO_DATE(bp.end_datetime, 'YYYY-MM-DD') as promo_end_date,
        DATE_PART('year', bp.start_datetime) as event_year,
        DATE_PART('month', bp.start_datetime) as event_month,
        (CASE 
            WHEN bp.promotion_key IS NULL THEN 'NO_PROMOTION'
            -- tier 1
            WHEN UPPER(bp.promotion_internal_title) LIKE '%BSS%' 
                OR UPPER(bp.promotion_internal_title) LIKE '%BIG SPRING SALE%' 
                THEN 'BSS'
            WHEN (DATE_PART('month', bp.start_datetime) = 7 
                OR (DATE_PART('month', bp.start_datetime) = 6 
                    AND DATE_PART('day', bp.start_datetime) >= 25))
                AND (
                    UPPER(bp.promotion_internal_title) LIKE '%PRIME%DAY%'
                    OR UPPER(bp.promotion_internal_title) LIKE '%PD%' 
                    OR UPPER(bp.promotion_internal_title) LIKE '%PEBD%' 
                )
                THEN 'PRIME DAY'
            WHEN UPPER(bp.promotion_internal_title) LIKE '%PBDD%' THEN 'PBDD'
            WHEN UPPER(bp.promotion_internal_title) LIKE '%BF%'
                OR UPPER(bp.promotion_internal_title) LIKE '%BLACK%FRIDAY%' THEN 'BLACK FRIDAY'
            WHEN UPPER(bp.promotion_internal_title) LIKE '%CYBER%MONDAY%'
                OR UPPER(bp.promotion_internal_title) LIKE '%CM%' THEN 'CYBER MONDAY'
            WHEN UPPER(bp.promotion_internal_title) LIKE '%BOXING WEEK%'
                OR UPPER(bp.promotion_internal_title) LIKE '%BOXING DAY%' THEN 'BOXING WEEK'
            WHEN UPPER(bp.promotion_internal_title) LIKE '%T5%'
                OR UPPER(bp.promotion_internal_title) LIKE '%T11%'
                OR UPPER(bp.promotion_internal_title) LIKE '%T12%' THEN 'T5/11/12'
            -- tier 1.5
            WHEN UPPER(bp.promotion_internal_title) LIKE '%BACK%TO%SCHOOL%' THEN 'BACK TO SCHOOL' 
            WHEN UPPER(bp.promotion_internal_title) LIKE '%BACK%TO%UNIVERSITY%' THEN 'BACK TO UNIVERSITY'
            -- tier 2-3
            WHEN bp.promotion_key IS NOT NULL THEN 'OTHER_PROMOTION'
            -- fallback
            ELSE 'NO_PROMOTION'
        END) as event_name
    FROM base_promos bp
);


/* Step 3: Create promotion details with prioritization */
DROP TABLE IF EXISTS promotion_details;
CREATE TEMP TABLE promotion_details AS (
    WITH event_priority AS (
        SELECT 
            region_id,
            marketplace_key,
            product_group_key,
            asin,
            promotion_key,
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
    )
    SELECT DISTINCT
        region_id,
        marketplace_key,
        product_group_key,
        asin,
        promotion_key,
        event_name,
        event_year,
        promo_start_date,
        promo_end_date
    FROM (
        SELECT 
            region_id,
            marketplace_key,
            product_group_key,
            asin,
            event_name,
            event_year,
            promotion_key,
            promo_start_date,
            promo_end_date,
            ROW_NUMBER() OVER (
                PARTITION BY asin
                ORDER BY event_priority_order, promo_start_date
            ) as event_rank
        FROM event_priority
    ) ranked
    WHERE event_rank = 1
);


-- Step 4: Find the most common start/end dates for each event
DROP TABLE IF EXISTS event_standards;
CREATE TEMP TABLE event_standards AS (
    WITH event_counts AS (
        SELECT 
            region_id,
            marketplace_key,
            product_group_key,
            event_name,
            event_year,
            -- event_month,
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
            region_id,
            marketplace_key,
            product_group_key,
            event_name,
            event_year,
            -- event_month,
            promo_start_date,
            promo_end_date
    )
    SELECT 
        region_id,
        marketplace_key,
        product_group_key,
        event_name,
        event_year,
        -- event_month,
        promo_start_date,
        promo_end_date,
        frequency
    FROM event_counts
    WHERE rn = 1
        AND frequency >= 3  -- Only keep patterns used by at least 3 promotions
);


-- Step 5: Final consolidated promotions
DROP TABLE IF EXISTS consolidated_promos;
CREATE TEMP TABLE consolidated_promos AS (
    SELECT 
        p.region_id,
        p.marketplace_key,
        p.product_group_key,
        p.asin,
        -- p.customer_shipment_item_id,
        p.event_name,
        DATE_PART('year', p.promo_start_date) as event_year,
        -- Use the standard event month for consistency
        -- Use standard dates if they exist, otherwise use original dates
        COALESCE(e.promo_start_date, p.promo_start_date) as promo_start_date,
        COALESCE(e.promo_end_date, p.promo_end_date) as promo_end_date
    FROM promotion_details p
        LEFT JOIN event_standards e
        ON p.event_name = e.event_name
        AND DATE_PART('year', p.promo_start_date) = e.event_year
);


/*-----------------------------------------------------------
                      assemble
-------------------------------------------------------------*/
/* Step 6: Add event info back output */
-- if deal price != price, then not an event purchase
DROP TABLE IF EXISTS orders_event;
CREATE TEMP TABLE orders_event AS (
    SELECT 
        bo.asin,
        bo.item_name,
        bo.order_date,
        -- bo.gl_product_group, 
        -- bo.company_code,
        bo.price,
        (CASE 
            WHEN bp.promotion_pricing_amount != bo.price THEN 'BAU'
            WHEN pd.event_name IS NULL OR pd.event_name = 'NO_PROMOTION' THEN 'BAU'
            ELSE pd.event_name 
        END) as event_name,
        -- COALESCE(ROUND(bp.promotion_pricing_amount,2), 0) as promotion_pricing_amount,
        (CASE
            WHEN bp.promotion_pricing_amount != bo.price THEN 0
            ELSE COALESCE(ROUND(bp.current_discount_percent,2), 0) 
        END) as deal_discount_percent,
        CAST(SUM(bo.shipped_units) as int) as shipped_units,
        ROUND(SUM(bo.revenue_share_amt), 2) as revenue
    FROM base_orders bo
        LEFT JOIN consolidated_promos pd
            on bo.asin = pd.asin
            AND bo.order_date BETWEEN pd.promo_start_date AND pd.promo_end_date
        LEFT JOIN base_promos bp
            on pd.asin = bp.asin
            AND bo.order_date BETWEEN bp.start_datetime AND bp.end_datetime
    GROUP BY 
        bo.asin,
        bo.item_name,
        bo.order_date,
        -- bo.gl_product_group, 
        -- bo.company_code,
        bo.price,
        pd.event_name,
        bp.promotion_pricing_amount,
        bp.current_discount_percent
);


-- SELECT * FROM orders_event;
