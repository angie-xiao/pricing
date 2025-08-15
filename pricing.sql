
/*******#********************
Base Orders Query
- Includes only retail merchant orders with shipped units > 0 
- Excludes cancelled or fraudulent orders
- Filters for last 365 days 
***************************/
DROP TABLE IF EXISTS base_orders;
CREATE TEMP TABLE base_orders AS (
    SELECT 
        o.marketplace_id,
        maa.gl_product_group,
        v.company_code,
        o.asin,
        maa.item_name,
        TO_DATE(o.order_datetime, 'YYYY-MM-DD') as order_date,
        SUM(COALESCE(o.shipped_units),0) as shipped_units,
        SUM(COALESCE(cp.revenue_share_amt, 0)) as revenue_share_amt,
        ROUND(
            (
                CASE 
                WHEN SUM(COALESCE(o.shipped_units, 0)) = 0 THEN NULL
                ELSE SUM(COALESCE(cp.revenue_share_amt, 0)) / SUM(COALESCE(o.shipped_units, 0))
                END
            )
        , 2) as asp
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
            BETWEEN TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD') - interval '365 days'
            AND TO_DATE('{RUN_DATE_YYYY-MM-DD}', 'YYYY-MM-DD')
        AND o.order_condition != 6
    GROUP BY 
        maa.gl_product_group,
        v.company_code,
        o.asin,
        maa.item_name,
        TO_DATE(o.order_datetime, 'YYYY-MM-DD')
);
