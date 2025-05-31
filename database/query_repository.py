"""
Query repository for the sales database.
Encapsulates database queries with proper caching.
"""
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
from .query_executor import QueryExecutor

logger = logging.getLogger(__name__)


class QueryRepository:
    """Repository for all database queries."""
    
    def __init__(self, query_executor: QueryExecutor):
        """
        Initialize with a query executor.
        
        Args:
            query_executor: Database query executor
        """
        self.executor = query_executor

    
    def get_monthly_sales_trend(self) -> pd.DataFrame:
        """
        Get monthly sales trend over time.
        
        Returns:
            DataFrame with monthly sales trend data
        """
        query = """
        SELECT 
            DATE_FORMAT(created_at, '%Y-%m') AS month,
            COUNT(entity_id) AS order_count,
            SUM(base_total_paid) AS total_revenue
        FROM 
            sales_flat_order
        WHERE 
            state = 'complete'
        GROUP BY 
            DATE_FORMAT(created_at, '%Y-%m')
        ORDER BY 
            month
        """
        
        try:
            return self.executor.execute_query(query, cache_key="monthly_sales_trend")
        except Exception as e:
            logger.error(f"Error getting monthly sales trend: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['month', 'order_count', 'total_revenue'])
    
    def get_category_seasonality(self, category_attribute_id: int = 80) -> pd.DataFrame:
        """
        Get seasonal patterns by product category.
        
        Args:
            category_attribute_id: ID of the category attribute in database
            
        Returns:
            DataFrame with category seasonality data
        """
        query = """
        SELECT 
            cat.value AS category_name,
            DATE_FORMAT(sfo.created_at, '%Y-%m') AS month,
            COUNT(DISTINCT sfo.entity_id) AS order_count,
            SUM(sfoi.qty_ordered) AS units_sold,
            SUM(sfoi.base_row_total) AS category_revenue
        FROM 
            sales_flat_order sfo
        JOIN 
            sales_flat_order_item sfoi ON sfo.entity_id = sfoi.order_id
        JOIN 
            catalog_product_entity_int cpei ON sfoi.product_id = cpei.entity_id
        JOIN 
            eav_attribute_option_value cat ON cpei.value = cat.option_id
        WHERE 
            cpei.attribute_id = %s
            AND sfo.state = 'complete'
        GROUP BY 
            cat.value, DATE_FORMAT(sfo.created_at, '%Y-%m')
        ORDER BY 
            category_name, month
        """
        
        try:
            params = (category_attribute_id,)
            return self.executor.execute_query(query, params, cache_key=f"category_seasonality_{category_attribute_id}")
        except Exception as e:
            logger.error(f"Error getting category seasonality: {str(e)}")
            return pd.DataFrame(columns=['category_name', 'month', 'order_count', 'units_sold', 'category_revenue'])
    
    def get_promotion_impact(self) -> pd.DataFrame:
        """
        Get impact of promotions on sales volume.
        
        Returns:
            DataFrame with promotion impact data
        """
        query = """
        SELECT 
            cr.name AS promotion_name,
            COUNT(DISTINCT sfo.entity_id) AS order_count,
            SUM(sfo.base_total_paid) AS total_revenue,
            SUM(sfo.base_discount_amount) AS total_discount,
            AVG(sfo.base_discount_amount / sfo.base_subtotal) AS avg_discount_percentage,
            SUM(sfoi.qty_ordered) AS units_sold
        FROM 
            sales_flat_order sfo
        JOIN 
            sales_flat_order_item sfoi ON sfo.entity_id = sfoi.order_id
        LEFT JOIN 
            catalogrule cr ON FIND_IN_SET(cr.rule_id, sfoi.applied_rule_ids)
        WHERE 
            sfo.state = 'complete'
            AND sfo.created_at BETWEEN cr.from_date AND IFNULL(cr.to_date, CURRENT_DATE)
        GROUP BY 
            cr.name
        ORDER BY 
            total_revenue DESC
        """
        
        try:
            logger.info("Running promotion impact")
            return self.executor.execute_query(query, cache_key="promotion_impact")
        except Exception as e:
            logger.error(f"Error getting promotion impact: {str(e)}")
            return pd.DataFrame(columns=['promotion_name', 'order_count', 'total_revenue', 
                                       'total_discount', 'avg_discount_percentage', 'units_sold'])
    
    def get_sales_by_day_hour(self) -> pd.DataFrame:
        """
        Get sales variation by day of week and hour of day.
        
        Returns:
            DataFrame with sales by day and hour
        """
        query = """
        SELECT 
            DAYNAME(created_at) AS day_of_week,
            HOUR(created_at) AS hour_of_day,
            COUNT(entity_id) AS order_count,
            SUM(base_total_paid) AS total_revenue,
            AVG(base_total_paid) AS avg_order_value
        FROM 
            sales_flat_order
        WHERE 
            state = 'complete'
        GROUP BY 
            DAYNAME(created_at), HOUR(created_at)
        ORDER BY 
            FIELD(day_of_week, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'),
            hour_of_day
        """
        
        try:
            return self.executor.execute_query(query, cache_key="sales_by_day_hour")
        except Exception as e:
            logger.error(f"Error getting sales by day and hour: {str(e)}")
            return pd.DataFrame(columns=['day_of_week', 'hour_of_day', 'order_count', 'total_revenue', 'avg_order_value'])
    
    
    def get_campaign_timings(self) -> pd.DataFrame:
        """
        Get campaign/promotion timings.
        
        Returns:
            DataFrame with campaign timing data
        """
        query = """
        SELECT 
            rule_id,
            name AS campaign_name,
            from_date,
            to_date,
            discount_amount,
            DATEDIFF(IFNULL(to_date, CURRENT_DATE), from_date) AS duration_days,
            is_active
        FROM 
            catalogrule
        ORDER BY 
            from_date DESC
        """
        
        try:
            return self.executor.execute_query(query, cache_key="campaign_timings")
        except Exception as e:
            logger.error(f"Error getting campaign timings: {str(e)}")
            return pd.DataFrame(columns=['rule_id', 'campaign_name', 'from_date', 'to_date', 
                                       'discount_amount', 'duration_days', 'is_active'])
    
       
    def get_enhanced_category_item_sales(self, category_ids=[694, 685]) -> pd.DataFrame:
        """
        Get sales data for specific product categories.
        
        Args:
            category_ids: List of category IDs to include in the report (default: [694, 685])
            
        Returns:
            DataFrame with category sales data
        """
        category_ids_str = ', '.join(map(str, category_ids))
        
        query = f"""
        SELECT 
            soi.sku, 
            cpei.value AS category_num, 
            soi.item_id, 
            soi.order_id, 
            DATE(soi.created_at) AS order_date, 
            soi.qty_ordered, 
            soi.base_price, 
            soi.base_discount_amount, 
            soi.base_row_total_incl_tax, 
            soi.applied_rule_ids
        FROM 
            sales_flat_order sfo 
        JOIN 
            sales_flat_order_item soi ON sfo.entity_id = soi.order_id 
        JOIN 
            catalog_product_entity cpe ON soi.product_id = cpe.entity_id 
        JOIN 
            catalog_product_entity_int cpei ON cpe.entity_id = cpei.entity_id 
        JOIN 
            eav_attribute_option_value aov ON cpei.value = aov.option_id 
        WHERE 
            sfo.increment_id NOT LIKE 'EBAY%' 
            AND sfo.increment_id NOT LIKE 'AMZ%' 
            AND cpei.attribute_id = 168 
            AND aov.store_id = 0 
            AND cpei.value IN ({category_ids_str})
        """
        
        try:
            logger.info("Running category sales report")
            cache_key = f"category_item_sales_with_rule_ids_{'-'.join(map(str, sorted(category_ids)))}"
            return self.executor.execute_query(query, cache_key=cache_key)
        except Exception as e:
            logger.error(f"Error getting category sales data: {str(e)}")
            return pd.DataFrame(columns=['sku', 'category_num', 'item_id', 'order_id', 
                                        'order_date', 'qty_ordered', 'base_price', 
                                        'base_discount_amount', 'base_row_total_incl_tax', 'applied_rule_ids'])
        
    def get_number_of_products_2023(self, category_ids=[694, 685]) -> pd.DataFrame:
        """
        Get product counts for 2023.
        
        Args:
            category_ids: List of category IDs to include in the report (default: [694, 685])
            
        Returns:
            DataFrame with product count data
        """
        category_ids_str = ', '.join(map(str, category_ids))
        
        query = f"""
        WITH product_sales_2023 AS (
            SELECT 
                cpei.value AS category_num,
                cpei.entity_id AS product_id,
                COUNT(DISTINCT sfo.entity_id) AS order_count
            FROM catalog_product_entity_int cpei
            LEFT JOIN sales_flat_order_item soi ON cpei.entity_id = soi.product_id
            LEFT JOIN sales_flat_order sfo ON 
                soi.order_id = sfo.entity_id 
                AND sfo.increment_id NOT LIKE 'EBAY%' 
                AND sfo.increment_id NOT LIKE 'AMZ%'
                AND YEAR(sfo.created_at) = 2023
            WHERE 
                cpei.attribute_id = 168 
                AND cpei.value IN ({category_ids_str})
                /* Only include products that existed in 2023 */
                AND EXISTS (
                    SELECT 1 FROM catalog_product_entity cpe 
                    WHERE cpe.entity_id = cpei.entity_id 
                    AND YEAR(cpe.created_at) <= 2023
                )
            GROUP BY cpei.value, cpei.entity_id
        )
        SELECT 
            category_num,
            COUNT(DISTINCT product_id) AS total_products,
            COUNT(DISTINCT CASE WHEN order_count > 0 THEN product_id END) AS sold_products,
            COUNT(DISTINCT CASE WHEN order_count = 1 THEN product_id END) AS single_order_products,
            COUNT(DISTINCT CASE WHEN order_count = 0 THEN product_id END) AS unsold_products
        FROM product_sales_2023
        GROUP BY category_num;
        """
        
        try:
            logger.info("Running product counts 2023 report")
            cache_key = f"products_2023_{'-'.join(map(str, sorted(category_ids)))}"
            return self.executor.execute_query(query, cache_key=cache_key)
        except Exception as e:
            logger.error(f"Error getting 2023 product data: {str(e)}")
            return pd.DataFrame(columns=['category_num', 'total_products', 'sold_products', 
                                        'single_order_products', 'unsold_products'])


    def get_number_of_products_2024(self, category_ids=[694, 685]) -> pd.DataFrame:
        """
        Get product counts for 2024.
        
        Args:
            category_ids: List of category IDs to include in the report (default: [694, 685])
            
        Returns:
            DataFrame with product count data
        """
        category_ids_str = ', '.join(map(str, category_ids))
        
        query = f"""
        WITH product_sales_2024 AS (
            SELECT 
                cpei.value AS category_num,
                cpei.entity_id AS product_id,
                COUNT(DISTINCT sfo.entity_id) AS order_count
            FROM catalog_product_entity_int cpei
            LEFT JOIN sales_flat_order_item soi ON cpei.entity_id = soi.product_id
            LEFT JOIN sales_flat_order sfo ON 
                soi.order_id = sfo.entity_id 
                AND sfo.increment_id NOT LIKE 'EBAY%' 
                AND sfo.increment_id NOT LIKE 'AMZ%'
                AND YEAR(sfo.created_at) = 2024
            WHERE 
                cpei.attribute_id = 168 
                AND cpei.value IN ({category_ids_str})
                /* Only include products that existed in 2024 */
                AND EXISTS (
                    SELECT 1 FROM catalog_product_entity cpe 
                    WHERE cpe.entity_id = cpei.entity_id 
                    AND YEAR(cpe.created_at) <= 2024
                )
            GROUP BY cpei.value, cpei.entity_id
        )
        SELECT 
            category_num,
            COUNT(DISTINCT product_id) AS total_products,
            COUNT(DISTINCT CASE WHEN order_count > 0 THEN product_id END) AS sold_products,
            COUNT(DISTINCT CASE WHEN order_count = 1 THEN product_id END) AS single_order_products,
            COUNT(DISTINCT CASE WHEN order_count = 0 THEN product_id END) AS unsold_products
        FROM product_sales_2024
        GROUP BY category_num;
        """
        
        try:
            logger.info("Running product counts 2024 report")
            cache_key = f"products_2024_{'-'.join(map(str, sorted(category_ids)))}"
            return self.executor.execute_query(query, cache_key=cache_key)
        except Exception as e:
            logger.error(f"Error getting 2024 product data: {str(e)}")
            return pd.DataFrame(columns=['category_num', 'total_products', 'sold_products', 
                                        'single_order_products', 'unsold_products'])
        
    def get_cross_year_product_metrics(self, years=[2023, 2024], category_ids=[694, 685]) -> pd.DataFrame:
        """
        Get comprehensive product metrics across specified years.
        
        This query tracks products across multiple years to properly categorize them as:
        - Sold multiple times (across all specified years)
        - Sold exactly once (in any of the specified years)
        - Never sold (in any of the specified years)
        
        Args:
            years: List of years to analyze (default: [2023, 2024])
            category_ids: List of category IDs to include (default: [694, 685])
                
        Returns:
            DataFrame with cross-year product metrics
        """
        years_str = ', '.join(map(str, years))
        category_ids_str = ', '.join(map(str, category_ids))
        
        query = f"""
        WITH product_base AS (
            -- Get all products that existed by the last year in our analysis
            SELECT 
                cpei.value AS category_num,
                cpei.entity_id AS product_id
            FROM 
                catalog_product_entity_int cpei
            WHERE 
                cpei.attribute_id = 168 
                AND cpei.value IN ({category_ids_str})
                -- Only include products that existed by the last year in our analysis
                AND EXISTS (
                    SELECT 1 FROM catalog_product_entity cpe 
                    WHERE cpe.entity_id = cpei.entity_id 
                    AND YEAR(cpe.created_at) <= {max(years)}
                )
        ),
        product_sales AS (
            -- Get sales data for these products across all specified years
            SELECT 
                pb.category_num,
                pb.product_id,
                YEAR(sfo.created_at) AS sales_year,
                COUNT(DISTINCT sfo.entity_id) AS order_count
            FROM 
                product_base pb
            LEFT JOIN 
                sales_flat_order_item soi ON pb.product_id = soi.product_id
            LEFT JOIN 
                sales_flat_order sfo ON 
                    soi.order_id = sfo.entity_id 
                    AND sfo.increment_id NOT LIKE 'EBAY%' 
                    AND sfo.increment_id NOT LIKE 'AMZ%'
                    AND YEAR(sfo.created_at) IN ({years_str})
            GROUP BY 
                pb.category_num, pb.product_id, YEAR(sfo.created_at)
        ),
        product_sales_aggregated AS (
            -- Aggregate sales across all years for each product
            SELECT
                category_num,
                product_id,
                SUM(order_count) AS total_orders
            FROM
                product_sales
            GROUP BY
                category_num, product_id
        )
        -- Final categorization
        SELECT 
            category_num,
            COUNT(DISTINCT product_id) AS total_products,
            COUNT(DISTINCT CASE WHEN total_orders > 1 THEN product_id END) AS multiple_order_products,
            COUNT(DISTINCT CASE WHEN total_orders = 1 THEN product_id END) AS single_order_products,
            COUNT(DISTINCT CASE WHEN total_orders = 0 THEN product_id END) AS unsold_products
        FROM 
            product_sales_aggregated
        GROUP BY 
            category_num;
        """
        
        try:
            logger.info(f"Running cross-year product metrics for years: {years}")
            cache_key = f"cross_year_products_{'-'.join(map(str, sorted(years)))}_{'-'.join(map(str, sorted(category_ids)))}"
            return self.executor.execute_query(query, cache_key=cache_key)
        except Exception as e:
            logger.error(f"Error getting cross-year product metrics: {str(e)}")
            return pd.DataFrame(columns=['category_num', 'total_products', 'multiple_order_products', 
                                    'single_order_products', 'unsold_products'])
        
    def get_promotion_duration_stats(self) -> pd.DataFrame:
        """
        Get promotion duration statistics by year and combined periods.
        
        Returns:
            DataFrame with promotion duration statistics
        """
        query = """
        WITH date_ranges AS (
            SELECT '2023-01-01' AS start_date, '2023-12-31' AS end_date, '2023' AS year
            UNION ALL
            SELECT '2024-01-01', '2024-12-31', '2024'
            UNION ALL
            SELECT '2023-01-01', '2024-12-31', '2023-2024'
        )
        SELECT 
            dr.year,
            COUNT(DISTINCT cr.rule_id) AS promotion_count,
            ROUND(AVG(
                DATEDIFF(
                    LEAST(cr.to_date, dr.end_date), 
                    GREATEST(cr.from_date, dr.start_date)
                ) + 1
            ), 2) AS average_duration_days,
            MAX(
                DATEDIFF(
                    LEAST(cr.to_date, dr.end_date), 
                    GREATEST(cr.from_date, dr.start_date)
                ) + 1
            ) AS max_duration_days,
            MIN(
                DATEDIFF(
                    LEAST(cr.to_date, dr.end_date), 
                    GREATEST(cr.from_date, dr.start_date)
                ) + 1
            ) AS min_duration_days
        FROM 
            catalogrule cr,
            date_ranges dr
        WHERE 
            cr.to_date >= dr.start_date AND 
            cr.from_date <= dr.end_date
        GROUP BY
            dr.year
        ORDER BY
            CASE dr.year 
                WHEN '2023-2024' THEN 3 
                WHEN '2023' THEN 1 
                WHEN '2024' THEN 2 
            END;
        """
        
        try:
            return self.executor.execute_query(query, cache_key="promotion_duration_stats")
        except Exception as e:
            logger.error(f"Error getting promotion duration statistics: {str(e)}")
            return pd.DataFrame(columns=['year', 'promotion_count', 'average_duration_days', 
                                        'max_duration_days', 'min_duration_days'])

    def get_promotion_details(self) -> pd.DataFrame:
        """
        Get detailed information about promotions in 2023-2024.
        
        Returns:
            DataFrame with promotion details
        """
        query = """
        WITH date_range AS (
        SELECT 
            '2023-01-01' AS start_date,
            '2024-12-31' AS end_date
        )
        SELECT 
            cr.rule_id,
            cr.name,
            cr.from_date,
            cr.to_date,
            cr.discount_amount,
            cr.ctb_discount_amount,
            cr.ctb_units_used,
            cr.ctb_limit_units,
            cr.ctb_limit_discount,
            cr.ctb_discount_used
        FROM 
            catalogrule cr, 
            date_range
        WHERE 
            cr.to_date >= date_range.start_date AND 
            cr.from_date <= date_range.end_date
        ORDER BY 
            cr.from_date DESC;
        """
        
        try:
            return self.executor.execute_query(query, cache_key="promotion_details")
        except Exception as e:
            logger.error(f"Error getting promotion details: {str(e)}")
            return pd.DataFrame(columns=['rule_id', 'name', 'from_date', 'to_date', 
                                        'discount_amount', 'ctb_discount_amount', 'ctb_units_used',
                                        'ctb_limit_units', 'ctb_limit_discount', 'ctb_discount_used'])
        
    def get_enhanced_category_item_sales(self, category_ids=[694, 685]) -> pd.DataFrame:
        """
        Get enhanced sales data for specific product categories with additional customer and product info.
        
        Args:
            category_ids: List of category IDs to include in the report (default: [694, 685])
            
        Returns:
            DataFrame with enhanced category sales data including customer info and brand_id
        """
        category_ids_str = ', '.join(map(str, category_ids))
        
        # Clear previous cache for this query type
        old_cache_key = f"category_item_sales_{'-'.join(map(str, sorted(category_ids)))}"
        cache_key = f"enhanced_category_item_sales_{'-'.join(map(str, sorted(category_ids)))}"
        self.executor.cache.invalidate(old_cache_key)
        
        # Using a fixed salt for consistent hashing
        email_salt = "D3xu5Gh6J9pLqP2s"
        
        query = f"""
        SELECT
            soi.sku,
            cpei_category.value AS category_num,
            soi.item_id,
            soi.order_id,
            DATE(sfo.created_at) AS order_date,
            soi.qty_ordered,
            soi.base_price,
            soi.base_discount_amount,
            soi.base_row_total_incl_tax,
            soi.applied_rule_ids,
            -- Additional customer information
            sfo.customer_is_guest,
            sfo.customer_group_id,
            SHA2(CONCAT(sfo.customer_email, '{email_salt}'), 256) AS hashed_customer_email,
            -- Brand ID (attribute_id = 81)
            cpei_attr81.value AS brand_id
        FROM
            sales_flat_order sfo
        JOIN
            sales_flat_order_item soi ON sfo.entity_id = soi.order_id
        JOIN
            catalog_product_entity cpe ON soi.product_id = cpe.entity_id
        JOIN
            catalog_product_entity_int cpei_category ON cpe.entity_id = cpei_category.entity_id
        JOIN
            eav_attribute_option_value aov ON cpei_category.value = aov.option_id
        -- Left join for the brand attribute (81)
        LEFT JOIN
            catalog_product_entity_int cpei_attr81 ON cpe.entity_id = cpei_attr81.entity_id AND cpei_attr81.attribute_id = 81
        WHERE
            sfo.increment_id NOT LIKE 'EBAY%'
            AND sfo.increment_id NOT LIKE 'AMZ%'
            AND cpei_category.attribute_id = 168
            AND aov.store_id = 0
            AND cpei_category.value IN ({category_ids_str})
        """
        
        try:
            logger.info("Running enhanced category sales report")
            return self.executor.execute_query(query, cache_key=cache_key)
        except Exception as e:
            logger.error(f"Error getting enhanced category sales data: {str(e)}")
            return pd.DataFrame(columns=['sku', 'category_num', 'item_id', 'order_id',
                                    'order_date', 'qty_ordered', 'base_price',
                                    'base_discount_amount', 'base_row_total_incl_tax', 
                                    'applied_rule_ids', 'customer_is_guest', 
                                    'customer_group_id', 'hashed_customer_email',
                                    'brand_id'])