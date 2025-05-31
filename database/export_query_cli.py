#!/usr/bin/env python
"""
Command-line tool for exporting database query results to files.

Usage:
    python export_query_cli.py --query "SELECT * FROM sales_flat_order LIMIT 10" --format csv
    python export_query_cli.py --query-file queries/my_query.sql --format excel
    python export_query_cli.py --repo-method get_monthly_sales_trend --format json
"""
import argparse
import sys
import os
import getpass
import logging
from pathlib import Path
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from database import DatabaseConnection, QueryExecutor, QueryRepository
from database.query_exporter import QueryExporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export database query results to files")

    parser.add_argument(
        "--host",
        default="10.7.0.111",
        help="Database host (default: 10.7.0.111)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3306,
        help="Database port (default: 3306)"
    )
    parser.add_argument(
        "--database",
        default="Care2Beauty",
        help="Database name (default: Care2Beauty)"
    )
    parser.add_argument(
        "--username",
        help="Database username (will prompt if not provided)"
    )
    parser.add_argument(
        "--password",
        help="Database password (will prompt if not provided)"
    )
    
    # Query source (one of these must be provided)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        help="SQL query to execute"
    )
    query_group.add_argument(
        "--query-file",
        help="File containing SQL query to execute"
    )
    query_group.add_argument(
        "--repo-method",
        help="Repository method to call (e.g., get_monthly_sales_trend)"
    )
    
    # Method arguments (for repository methods)
    parser.add_argument(
        "--method-args",
        help="JSON string with arguments for repository method (e.g., '{\"limit\": 5}')"
    )
    
    # Query parameters
    parser.add_argument(
        "--params",
        help="JSON string with query parameters (e.g., '[10, \"value\"]')"
    )
    
    # Export options
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "excel", "json", "html", "pickle"],
        help="Export format (default: csv)"
    )
    parser.add_argument(
        "--filename",
        help="Base filename for export (without extension)"
    )
    parser.add_argument(
        "--export-dir",
        default="exports",
        help="Directory to store exported files (default: exports)"
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include query metadata in the export (for CSV and Excel formats)"
    )
    parser.add_argument(
        "--include-index",
        action="store_true",
        help="Include DataFrame index in the export"
    )
    
    return parser.parse_args()


def get_query_from_file(file_path):
    """Read query from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading query file: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Get username if not provided
    username = args.username
    if not username:
        username = input("Enter database username: ")
    
    # Get password if not provided
    password = args.password
    if not password:
        password = getpass.getpass("Enter database password: ")
    
    try:
        # Set up database connection
        db_config = {
            "host": args.host,
            "port": args.port,
            "database": args.database
        }
        
        # Connect to database
        logger.info(f"Connecting to database {args.database} on {args.host}:{args.port}")
        db_connection = DatabaseConnection(db_config)
        db_connection.connect(username, password)
        
        # Create query executor and repository
        query_executor = QueryExecutor(db_connection)
        query_repo = QueryRepository(query_executor)
        
        # Create query exporter
        exporter = QueryExporter(export_dir=args.export_dir)
        
        # Determine query source and execute export
        if args.query:
            # Direct query string
            query = args.query
            params = json.loads(args.params) if args.params else None
            
            logger.info(f"Exporting results of custom query")
            export_path = exporter.export_query(
                query_executor,
                query,
                params=params,
                filename=args.filename,
                format=args.format,
                include_metadata=args.include_metadata,
                include_index=args.include_index
            )
            
        elif args.query_file:
            # Query from file
            query = get_query_from_file(args.query_file)
            params = json.loads(args.params) if args.params else None
            
            # Use filename from query file if not specified
            filename = args.filename
            if not filename:
                filename = os.path.splitext(os.path.basename(args.query_file))[0]
            
            logger.info(f"Exporting results of query from file: {args.query_file}")
            export_path = exporter.export_query(
                query_executor,
                query,
                params=params,
                filename=filename,
                format=args.format,
                include_metadata=args.include_metadata,
                include_index=args.include_index
            )
            
        elif args.repo_method:
            # Repository method
            method_args = json.loads(args.method_args) if args.method_args else {}
            
            logger.info(f"Exporting results of repository method: {args.repo_method}")
            export_path = exporter.export_repository_query(
                query_repo,
                args.repo_method,
                method_args=method_args,
                filename=args.filename,
                format=args.format,
                include_index=args.include_index
            )
        
        logger.info(f"Successfully exported data to: {export_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
        
    finally:
        # Disconnect from database
        if 'db_connection' in locals() and db_connection:
            db_connection.disconnect()


if __name__ == "__main__":
    main()