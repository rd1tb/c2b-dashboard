"""
Authentication utilities for the sales dashboard.
"""
import streamlit as st
from typing import Tuple, Optional
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from database import DatabaseConnection, QueryExecutor, QueryRepository


def display_login_form() -> None:
    """Display login form and authenticate user."""
    st.title("Sales Data Explorer")
    predefined_users = ["user1_gr2", "user2_gr2", "user3_gr2", "user4_gr2", "user5_gr2"]
    with st.form("login_form"):
        username = st.selectbox(
            "Select Username", 
            options=predefined_users,
            index=0
        )
        password = st.text_input("Database Password", type="password")
        submitted = st.form_submit_button("Connect to Database")
        if submitted and username and password:
            authenticate_user(username, password)


def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate user with database credentials.
    
    Args:
        username: Database username
        password: Database password
    
    Returns:
        True if authentication successful, False otherwise
    """
    if not username or not password:
        st.error("Username and password are required")
        return False
        
    try:
        db_connection = DatabaseConnection()
        db_connection.connect(username, password)
        query_executor = QueryExecutor(db_connection)
        query_repo = QueryRepository(query_executor)
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.db_connection = db_connection
        st.session_state.query_executor = query_executor
        st.session_state.query_repo = query_repo
        st.success("Connected successfully!")
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return False


def logout() -> None:
    """Log out the current user."""
    if 'db_connection' in st.session_state and st.session_state.db_connection:
        st.session_state.db_connection.disconnect()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def init_session_state() -> None:
    """Initialize session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "db_connection" not in st.session_state:
        st.session_state.db_connection = None
    if "query_executor" not in st.session_state:
        st.session_state.query_executor = None
    if "query_repo" not in st.session_state:
        st.session_state.query_repo = None