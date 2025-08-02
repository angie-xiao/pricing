
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
        o.customer_id,
        o.customer_shipment_item_id,
        TO_DATE(o.order_datetime, 'YYYY-MM-DD') as order_date,
        o.our_price, -- price shown on amazon
        o.shipped_units,
        maa.gl_product_group,
        maa.brand_name,
        maa.brand_code,
        COALESCE(cp.revenue_share_amt, 0) as revenue_share_amt,
        -- mam.dama_mfg_vendor_code as vendor_code,
        v.company_code,
        v.company_name
    FROM andes.booker.d_unified_cust_shipment_items o
        INNER JOIN andes.booker.d_mp_asin_attributes maa
            ON maa.asin = o.asin
            AND maa.marketplace_id = o.marketplace_id
            AND maa.region_id = o.region_id
            AND maa.gl_product_group IN 199
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
            BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '365 days'
            AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
        AND o.order_condition != 6
);


DROP TABLE IF EXISTS promotion_details;
CREATE TEMP TABLE promotion_details AS (
    SELECT DISTINCT
        f.customer_shipment_item_id,
        f.asin,
        p.promotion_key,
        TO_DATE(p.start_datetime, 'YYYY-MM-DD') as promo_start_date,
        TO_DATE(p.end_datetime, 'YYYY-MM-DD') as promo_end_date,
        p.promotion_internal_title,
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

            -- tier 2
            WHEN UPPER(p.promotion_internal_title) LIKE '%NYNY%' THEN 'NYNY'
            WHEN UPPER(p.promotion_internal_title) LIKE '%GIFTMANIA%' 
                OR UPPER(p.promotion_internal_title) LIKE '%GIFT%MANIA%' THEN 'GIFT MANIA'
            WHEN UPPER(p.promotion_internal_title) LIKE '%HALLOWEEN%' THEN 'HALLOWEEN'
            WHEN UPPER(p.promotion_internal_title) LIKE '%HOLIDAY%' THEN 'HOLIDAY'
            
            -- tier 3    
            WHEN UPPER(p.promotion_internal_title) LIKE '%LUNAR%NEW%YEAR%' THEN 'LUNAR NEW YEAR'
            WHEN UPPER(p.promotion_internal_title) LIKE '%DAILY%ESSENTIALS%' THEN 'DAILY ESSENTIALS'
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
            WHEN UPPER(p.promotion_internal_title) LIKE '%DIWALI%' THEN 'DIWALI'
            WHEN UPPER(p.promotion_internal_title) LIKE '%MOVEMBER%' THEN 'MOVEMBER'
            WHEN UPPER(p.promotion_internal_title) LIKE '%FLASH SALE%' THEN 'FLASH SALE'
            ELSE 'OTHER'
        END) as event_name
    FROM andes.pdm.fact_promotion_cp f
        JOIN 
            (
                SELECT 
                    promotion_key,
                    start_datetime,
                    end_datetime,
                    promotion_internal_title
                FROM andes.pdm.dim_promotion
                WHERE marketplace_key = 7
                    AND approval_status IN ('Approved', 'Scheduled')
                    AND promotion_type IN ('Best Deal', 'Deal of the Day', 'Lightning Deal', 'Event Deal')
                    -- exclude inventory health promos
                    AND UPPER(promotion_internal_title) NOT LIKE '%OIH%'
                    -- exclude lead in / out promos
                    AND UPPER(promotion_internal_title) NOT LIKE '%LEAD%IN%'
                    AND UPPER(promotion_internal_title) NOT LIKE '%LEAD%OUT%'
                    AND UPPER(promotion_internal_title) NOT LIKE '%LEADIN%'
                    AND UPPER(promotion_internal_title) NOT LIKE '%LEADOUT%'

                    AND TO_DATE(start_datetime, 'YYYY-MM-DD') 
                        BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '730 days'
                        AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
        ) p
        ON f.promotion_key = p.promotion_key
);


/*************************
Create unified base table combining orders and metrics
*************************/
DROP TABLE IF EXISTS unified_deal_base;
CREATE TEMP TABLE unified_deal_base AS (
    
    SELECT 
        -- Deal context
        d.asin,
        d.event_name,
        d.event_year,
 
        -- Order date
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
        b.our_price,
        b.shipped_units,
        b.revenue_share_amt

    FROM promotion_details d
        INNER join andes.pdm.DIM_PROMOTION_ASIN  pa
            on d.promotion_key = pa.promotion_key
            and d.asin = pa.asin
            and pa.marketplace_id = 7
            and pa.region_id = 1
        RIGHT JOIN base_orders b 
            ON d.asin = b.asin
            AND d.customer_shipment_item_id = b.customer_shipment_item_id
);