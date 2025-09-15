/* Step 1: Get base promotion information */
-- currently filtering for CA Pets
DROP TABLE IF EXISTS base_promos;
CREATE TEMP TABLE base_promos AS (
    SELECT DISTINCT
        p.region_id,
        p.marketplace_key,
        pa.product_group_key,
        pa.asin,
        p.promotion_key,
        p.promotion_internal_title,
        p.start_datetime,
        p.end_datetime,
        pa.promotion_pricing_amount
    FROM andes.pdm.dim_promotion p
        JOIN andes.pdm.dim_promotion_asin pa
            ON p.promotion_key = pa.promotion_key
            AND p.region_id = pa.region_id
            AND p.marketplace_key = pa.marketplace_key
    WHERE p.region_id = 1
        AND p.marketplace_key = 7
        AND TO_DATE(start_datetime, 'YYYY-MM-DD') <= TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '730 days'
        AND p.approval_status IN ('Approved', 'Scheduled')
        AND p.promotion_type IN ('Best Deal', 'Deal of the Day', 'Lightning Deal', 'Event Deal')
        AND UPPER(p.promotion_internal_title) NOT LIKE '%OIH%'
        AND pa.product_group_key=199  -- Pets
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
            ELSE 'OTHER'
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
        promo_start_date,
        promo_end_date
    FROM (
        SELECT 
            region_id,
            marketplace_key,
            product_group_key,
            asin,
            event_name,
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


/* Step 4: Identify T4W shipments */
-- First identify pre-deal shipments
DROP TABLE IF EXISTS t4w_shipments;
CREATE TEMP TABLE t4w_shipments AS (
    SELECT 
        o.asin,
        bp.region_id,
        bp.marketplace_key,
        bp.product_group_key,
        bp.start_datetime,
        o.customer_shipment_item_id,
        o.shipped_units as shipped_units_4w
    FROM andes.booker.d_unified_cust_shipment_items o
        INNER JOIN base_promos bp 
            ON o.asin = bp.asin
            AND o.region_id = bp.region_id
            AND o.marketplace_id = bp.marketplace_key  
            AND o.gl_product_group = bp.product_group_key
            AND o.order_datetime BETWEEN bp.start_datetime - interval '28 days' AND bp.start_datetime - interval '1 day'
    WHERE o.order_condition != 6                      
        AND o.shipped_units > 0
        AND o.is_retail_merchant = 'Y'
);


-- then calculate T4W asp
DROP TABLE IF EXISTS t4w;
CREATE TEMP TABLE t4w AS (
    SELECT 
        o.asin,
        o.marketplace_key,
        o.product_group_key,
        CAST(o.start_datetime AS DATE) AS start_datetime,
        COALESCE(
            SUM(CASE WHEN cp.revenue_share_amt IS NOT NULL THEN cp.revenue_share_amt ELSE 0 END) / 
            NULLIF(SUM(CASE WHEN o.shipped_units_4w IS NOT NULL THEN o.shipped_units_4w ELSE 0 END), 0),
        0) AS t4w_asp
    FROM andes.contribution_ddl.o_wbr_cp_na cp 
        INNER JOIN t4w_shipments o
        ON cp.customer_shipment_item_id = o.customer_shipment_item_id 
        AND cp.asin = o.asin
        AND o.marketplace_key = cp.marketplace_id
    GROUP BY
        o.asin,
        o.marketplace_key,
        o.product_group_key,
        CAST(o.start_datetime AS DATE) 
);

/* Step 5: Get all shipments */
-- First, create filtered shipment table
-- Currently filtering for CA Pets
DROP TABLE IF EXISTS filtered_shipments;
CREATE TEMP TABLE filtered_shipments AS (
    SELECT 
        o.asin,
        o.customer_shipment_item_id,
        o.marketplace_id,
        o.region_id,
        o.our_price as price,
        TO_DATE(o.order_datetime, 'YYYY-MM-DD') as order_date,
        o.shipped_units
    FROM andes.booker.d_unified_cust_shipment_items o
    WHERE o.region_id = 1
        AND o.marketplace_id = 7
        AND o.gl_product_group IN (199)
        AND o.order_datetime BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '730 days'
            AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
        AND o.order_condition != 6
        AND o.shipped_units > 0
        AND o.is_retail_merchant = 'Y'
);


-- Then create the final base_orders table
-- currently filtering for Boxiecat (BO92F)
DROP TABLE IF EXISTS base_orders;
CREATE TEMP TABLE base_orders AS (
    SELECT DISTINCT
        fs.asin,
        maa.item_name,
        fs.customer_shipment_item_id,
        fs.order_date,
        maa.gl_product_group,
        maa.brand_name,
        maa.brand_code,
        mam.dama_mfg_vendor_code as vendor_code,
        v.company_code,
        v.company_name,
        fs.shipped_units,
        fs.price,
        COALESCE(cp.revenue_share_amt, 0) as revenue_share_amt
    FROM filtered_shipments fs
        INNER JOIN andes.booker.d_mp_asin_attributes maa
            ON maa.asin = fs.asin
            AND maa.marketplace_id = fs.marketplace_id
            AND maa.region_id = fs.region_id
            AND maa.gl_product_group IS NOT NULL
        LEFT JOIN andes.contribution_ddl.o_wbr_cp_na cp
            ON fs.customer_shipment_item_id = cp.customer_shipment_item_id 
            AND fs.asin = cp.asin
            AND fs.marketplace_id = cp.marketplace_id
        LEFT JOIN andes.BOOKER.D_MP_ASIN_MANUFACTURER mam
            ON mam.asin = fs.asin
            AND mam.marketplace_id = 7
            AND mam.region_id = 1
        LEFT JOIN andes.roi_ml_ddl.VENDOR_COMPANY_CODES v
            ON v.vendor_code = mam.dama_mfg_vendor_code
    WHERE v.company_code='BO92F'  -- Boxiecat
    
);


/* Step 6: Add event info back output */
DROP TABLE IF EXISTS orders_event;
CREATE TEMP TABLE orders_event AS (
    SELECT 
        bp.asin,
        bo.item_name,
        bo.order_date,
        bo.shipped_units,
        bo.gl_product_group,
        bo.brand_name,
        bo.brand_code,
        bo.revenue_share_amt as revenue,
        bo.vendor_code,
        bo.company_code,
        bo.company_name,
        bo.price,
        -- COALESCE(t.t4w_asp,0) as pre_deal_price,
        COALESCE(pd.event_name, 'BAU') as event,
        (CASE 
            WHEN bp.promotion_pricing_amount IS NOT NULL AND t.t4w_asp > 0 
            THEN (t.t4w_asp - bp.promotion_pricing_amount)
            ELSE 0 
        END) as discount_amt
    FROM base_promos bp
        LEFT JOIN t4w t 
            ON bp.asin = t.asin
            and t.marketplace_key = bp.marketplace_key
            and t.product_group_key = bp.product_group_key
            and t.start_datetime = bp.start_datetime
        LEFT JOIN promotion_details pd 
            ON bp.asin = pd.asin
        LEFT JOIN base_orders bo
            ON bp.asin = bo.asin
    GROUP BY 
        bp.asin,
        bo.item_name,
        bo.order_date,
        bo.shipped_units,
        bo.gl_product_group,
        bo.brand_name,
        bo.brand_code,
        bo.revenue_share_amt,
        bo.vendor_code,
        bo.company_code,
        bo.company_name,
        bo.price,
        bp.promotion_pricing_amount
        t.t4w_asp,
        pd.event_name
);


/* Step 7: Add event info back output */
DROP TABLE IF EXISTS final_output;
CREATE TEMP TABLE final_output AS (
    SELECT 
        bo.asin,
        bo.item_name,
        bo.event,
        bo.vendor_code,
        bo.company_code,
        bo.company_name,
        bo.price,
        bo.discount_amt,
        -- bo.pre_deal_price,
        -- bo.gl_product_group,
        -- bo.brand_name,
        -- bo.brand_code,
        count(distinct bo.order_date) as days_sold_at_price,
        sum(bo.shipped_units) as shipped_units,
        sum(bo.revenue) as revenue
    FROM orders_event bo
    GROUP BY 
        bo.asin,
        bo.item_name,
        bo.order_date,
        bo.shipped_units,
        -- bo.pre_deal_price,
        -- bo.gl_product_group,
        -- bo.brand_name,
        -- bo.brand_code,
        bo.revenue,
        bo.vendor_code,
        bo.company_code,
        bo.company_name,
        bo.price,
        bo.event,
        bo.discount_amt
);


SELECT * FROM final_output;
