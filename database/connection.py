import os
import json
import mysql.connector
from typing import Dict, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection class for MySQL."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize database connection.
        
        Args:
            config: Configuration dictionary with connection parameters.
                   If None, default configuration will be used.
        """
        self.config = config or {
            "host": "10.7.0.111",
            "port": 3306,
            "database": "Care2Beauty"
        }
        self.connection = None
        
    def connect(self, username: str, password: str) -> None:
        """
        Connect to the database.
        
        Args:
            username: Database username
            password: Database password
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.config.get("host", "10.7.0.111"),
                port=self.config.get("port", 3306),
                database=self.config.get("database", "Care2Beauty"),
                user=username,
                password=password
            )
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Disconnected from database")
    
    def get_cursor(self, dictionary=True):
        """
        Get a database cursor.
        
        Args:
            dictionary: Whether to return results as dictionaries
            
        Returns:
            Database cursor
        """
        if not self.connection or not self.connection.is_connected():
            raise ConnectionError("Not connected to database")
        
        return self.connection.cursor(dictionary=dictionary)