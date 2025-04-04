"""
Query exporter module for database result export functionality.
Allows exporting database query results to various file formats.
"""
import pandas as pd
import os
import logging
from typing import Optional, Dict, Tuple, List, Union, Callable
from pathlib import Path
import json
import datetime

# Set up logging
logger = logging.getLogger(__name__)


class QueryExporter:
    """Export database query results to files."""
    
    def __init__(self, export_dir: str = "exports"):
        """
        Initialize the query exporter.
        
        Args:
            export_dir: Directory to store exported files
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, base_name: str, file_format: str) -> str:
        """
        Generate a unique filename with timestamp.
        
        Args:
            base_name: Base filename
            file_format: File extension/format
            
        Returns:
            Unique filename with timestamp
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Replace spaces and special characters with underscores
        clean_name = "".join(c if c.isalnum() else "_" for c in base_name)
        return f"{clean_name}_{timestamp}.{file_format}"
    
    def export_dataframe(
        self, 
        df: pd.DataFrame, 
        filename: str = None, 
        format: str = "csv",
        include_index: bool = False,
        **kwargs
    ) -> str:
        """
        Export a DataFrame to a file.
        
        Args:
            df: DataFrame to export
            filename: Filename (without extension, will be generated if None)
            format: Export format ('csv', 'excel', 'json', 'html', 'pickle')
            include_index: Whether to include DataFrame index in export
            **kwargs: Additional export options passed to pandas
            
        Returns:
            Path to the exported file
        """
        if df.empty:
            logger.warning("Cannot export empty DataFrame")
            return None
            
        # Generate base name if not provided
        base_name = filename or "query_export"
        
        # Generate full filename with timestamp
        full_filename = self._generate_filename(base_name, format)
        file_path = self.export_dir / full_filename
        
        try:
            # Export based on format
            if format.lower() == "csv":
                df.to_csv(file_path, index=include_index, **kwargs)
            elif format.lower() in ["excel", "xlsx"]:
                if not str(file_path).endswith(('.xls', '.xlsx')):
                    file_path = self.export_dir / f"{os.path.splitext(full_filename)[0]}.xlsx"
                df.to_excel(file_path, index=include_index, **kwargs)
            elif format.lower() == "json":
                df.to_json(file_path, orient=kwargs.get("orient", "records"), **kwargs)
            elif format.lower() == "html":
                df.to_html(file_path, index=include_index, **kwargs)
            elif format.lower() in ["pickle", "pkl"]:
                df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            logger.info(f"Exported DataFrame to {file_path}")
            return str(file_path)
                
        except Exception as e:
            logger.error(f"Error exporting DataFrame: {str(e)}")
            raise
    
    def export_query(
        self,
        query_executor,
        query: str,
        params: Tuple = None,
        filename: str = None,
        format: str = "csv",
        include_metadata: bool = False,
        **kwargs
    ) -> str:
        """
        Execute a custom query and export results to a file.
        
        Args:
            query_executor: QueryExecutor instance
            query: SQL query string
            params: Query parameters
            filename: Filename (without extension, will be generated if None)
            format: Export format ('csv', 'excel', 'json', 'html', 'pickle')
            include_metadata: Include query metadata in the export
            **kwargs: Additional export options
            
        Returns:
            Path to the exported file
        """
        try:
            # Execute query
            result_df = query_executor.execute_query(query, params)
            
            # Generate filename if not provided
            if filename is None:
                # Use first 30 chars of query as filename, cleaned up
                query_snippet = query.strip().split('\n')[0][:30]
                filename = "".join(c if c.isalnum() or c.isspace() else "_" for c in query_snippet)
            
            # Add metadata if requested
            if include_metadata and format.lower() in ["csv", "excel", "xlsx"]:
                metadata_df = pd.DataFrame([
                    ["Query", query.replace('\n', ' ')],
                    ["Parameters", str(params) if params else "None"],
                    ["Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ["Row Count", len(result_df)]
                ], columns=["Metadata", "Value"])
                
                # Add empty row as separator
                separator = pd.DataFrame([["", ""]], columns=metadata_df.columns)
                
                # Combine metadata with results
                combined_df = pd.concat([metadata_df, separator, result_df.reset_index(drop=True)])
                return self.export_dataframe(combined_df, filename=filename, format=format, **kwargs)
            
            return self.export_dataframe(result_df, filename=filename, format=format, **kwargs)
            
        except Exception as e:
            logger.error(f"Error exporting query: {str(e)}")
            raise
            
    def export_repository_query(
        self,
        query_repo,
        method_name: str,
        method_args: Dict = None,
        filename: str = None,
        format: str = "csv",
        **kwargs
    ) -> str:
        """
        Execute a query from the repository and export results to a file.
        
        Args:
            query_repo: QueryRepository instance
            method_name: Name of the repository method to call
            method_args: Arguments to pass to the repository method
            filename: Filename (without extension, will be generated if None)
            format: Export format ('csv', 'excel', 'json', 'html', 'pickle')
            **kwargs: Additional export options
            
        Returns:
            Path to the exported file
        """
        try:
            # Get the method from the repository
            if not hasattr(query_repo, method_name):
                raise ValueError(f"Method '{method_name}' not found in query repository")
                
            repo_method = getattr(query_repo, method_name)
            
            # Call the method with provided arguments
            method_args = method_args or {}
            result = repo_method(**method_args)
            
            # Use method name as filename if not provided
            if filename is None:
                filename = method_name
            
            # Check if result is a DataFrame, if not, convert it to one
            if not isinstance(result, pd.DataFrame):
                # For scalar values (int, float, string, etc.)
                if not hasattr(result, '__iter__') or isinstance(result, (str, bytes)):
                    # Create a simple one-row DataFrame with the result
                    result_df = pd.DataFrame({method_name: [result]})
                else:
                    # For other iterables (list, tuple, etc.)
                    result_df = pd.DataFrame(result)
            else:
                result_df = result
                
            return self.export_dataframe(result_df, filename=filename, format=format, **kwargs)
            
        except Exception as e:
            logger.error(f"Error exporting repository query: {str(e)}")
            raise