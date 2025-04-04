## Project Structure

```
lced-care2beauty
│
├── database/
│   ├── __init__.py
│   ├── connection.py        # Database connection management
│   ├── query_cache.py       # Query caching mechanism
│   ├── query_executor.py    # Executes queries with caching
│   ├── query_exporter.py    # Exports query results to files
│   ├── export_query_cli.py  # CLI tool for exporting query results
│   └── repository.py        # Query repository with all query definitions
│
├── dashboard/
│   ├── __init__.py
│   ├── streamlit_app.py     # Main Streamlit application entry point
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication utilities
│   │   └── visualizations.py # Plotting and visualization utilities
│   │
│   └── views/
│       ├── __init__.py
│       ├── sales_view.py    # Sales overview view
│       ├── product_view.py  # Product analysis view
│       └── promotion_view.py   # Promotion analysis view
```

## Features

- **Database Connection Management**: Simple MySQL database connection handling
- **Query Caching**: File-based caching to improve performance of repeated queries
- **Query Repository**: Centralized repository of all database queries
- **Query Export**: Export query results to various file formats (CSV, Excel, JSON, HTML)
- **Streamlit Dashboard**: Interactive data visualization with multiple views

## Getting Started

```
git clone https://github.com/rd1tb/lced-care2beauty.git
cd lced-care2beauty
```

### Installation

Install uv
```
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

```
```
# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

```
Install requirements
```
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Running the Dashboard

```
streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### Usage

1. Log in with your MySQL database credentials
2. Navigate through different views using the sidebar
3. Explore sales data through interactive visualizations
4. Use the "Refresh Data" button to clear cache and fetch latest data

### Exporting Query Results
The project includes a command-line tool for exporting query results to various file formats:

Export results of a custom SQL query to CSV
```
python database/export_query_cli.py --query "SELECT * FROM sales_flat_order LIMIT 10" --format csv
```

Export results of a query from a file to Excel
```
python database/export_query_cli.py --query-file queries/monthly_sales.sql --format excel
```

Export results of a repository method to JSON
```
python database/export_query_cli.py --repo-method get_monthly_sales_trend --format json
```

Include query metadata in the exported file
```
python database/export_query_cli.py --query "SELECT * FROM catalogrule" --format csv --include-metadata
```
Specify a custom filename and export directory
```
python database/export_query_cli.py --repo-method get_top_products --method-args '{"limit": 20}' --filename top_20_products --export-dir exports/products --format excel
```
To see all available options:
```
python database/export_query_cli.py --help
```

#### Available Export Formats

- CSV (```--format csv```)
- Excel (```--format excel```)
- JSON (```--format json```)
- HTML (```--format html```)
- Pickle (```--format pickle```)

## Extending the Project
### Adding New Requirements
```
uv add package-name
```

### Adding New Queries

Add new query methods to the `QueryRepository` class in `db/repository.py`:

```python
def get_new_data(self, param1, param2) -> pd.DataFrame:
    """
    Description of the query.
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        DataFrame with results
    """
    query = """
    SELECT 
        column1, column2
    FROM 
        table
    WHERE 
        condition = %s AND other_condition = %s
    """
    
    try:
        params = (param1, param2)
        return self.executor.execute_query(query, params, cache_key=f"new_data_{param1}_{param2}")
    except Exception as e:
        logger.error(f"Error getting new data: {str(e)}")
        return pd.DataFrame(columns=['column1', 'column2'])
```

### Adding New Dashboard Views

1. Create a new view file in `dashboard/views/`:

```python
# dashboard/views/new_view.py
import streamlit as st
from dashboard.utils import create_bar_chart, format_currency
from database import QueryRepository

def display_new_view(query_repo: QueryRepository):
    """Display new view."""
    st.header("New View")
    
    # Get data
    with st.spinner("Loading data..."):
        data = query_repo.get_new_data(param1, param2)
    
    # Display visualizations
    if not data.empty:
        st.subheader("Data Visualization")
        fig = create_bar_chart(data, 'x_column', 'y_column', 'Chart Title')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available.")
```

2. Update the `__init__.py` file in the views folder
3. Add the new view to the navigation in `dashboard/streamlit_app.py`