"""
Dashboard views module.
"""
from .sales_view import display_sales_view
from .product_view import display_product_view
from .promotion_view import display_promotion_view

__all__ = [
    'display_sales_view',
    'display_product_view',
    'display_promotion_view'
]