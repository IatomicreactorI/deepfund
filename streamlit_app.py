import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import pandas as pd
import json
import os

# Database interaction class (replace with your actual implementation if needed)
# For now, it's just a placeholder to avoid errors if called elsewhere unexpectedly.
class DeepfundDB:
    def __enter__(self):
        # Placeholder connection logic if needed
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Placeholder disconnection logic if needed
        pass
    def get_workflow_data(self):
        # This method will be replaced by loading from CSVs
        return {} 
    def get_latest_portfolio(self, config_id):
         # This method is no longer directly called but kept for placeholder
        return None
    def get_latest_decisions(self, portfolio_id):
        # This method is no longer called but kept for placeholder
        return []


# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .index-container {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .index-title {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .index-value {
        font-size: 1.1rem;
    }
    .negative {
        color: red;
    }
    .positive {
        color: green;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        white-space: pre;
        padding-top: 5px;
        padding-bottom: 5px;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    footer {
        display: none;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .chart-toggle {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
    .workflow-button {
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .time-period-tabs {
        display: flex;
        gap: 2px;
        background: #f1f3f4;
        border-radius: 4px;
        padding: 2px;
        width: fit-content;
    }
    .time-period-tab {
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        cursor: pointer;
    }
    .time-period-tab.active {
        background-color: #007BFF;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header Navigation (Simplified)
st.markdown('<div class="header">Deepfund</div>', unsafe_allow_html=True)
cols = st.columns(6)
with cols[0]:
    st.markdown('<div style="display: flex; align-items: center;"><img src="https://static.tradingview.com/static/images/logo-tradingview.svg" width="120px"></div>', unsafe_allow_html=True)
with cols[1]:
    st.text_input("🔍 Search (Ctrl+K)", placeholder="Search models, workflows, strategies...")
with cols[2]:
    st.markdown('<div style="text-align: center;">Products</div>', unsafe_allow_html=True)
with cols[3]:
    st.markdown('<div style="text-align: center;">Community</div>', unsafe_allow_html=True)
with cols[4]:
    st.markdown('<div style="text-align: center; color: #2962FF; font-weight: bold;">Markets</div>', unsafe_allow_html=True)
with cols[5]:
    st.markdown('<div style="text-align: center;">Brokers</div>', unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">Deepfund: The most advanced LLM finance arena</h1>', unsafe_allow_html=True)

# --- Data Loading and Processing ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_process_data(config_path='data/config_rows.csv', portfolio_path='data/portfolio_rows.csv'):
    """Loads data from CSVs, processes it, and merges config with latest portfolio."""
    try:
        if not os.path.exists(config_path):
            st.error(f"Config file not found at: {config_path}")
            return {}
        if not os.path.exists(portfolio_path):
            st.error(f"Portfolio file not found at: {portfolio_path}")
            return {}
            
        config_df = pd.read_csv(config_path)
        portfolio_df = pd.read_csv(portfolio_path)

        # Data Cleaning and Type Conversion
        config_df['updated_at'] = pd.to_datetime(config_df['updated_at'])
        portfolio_df['updated_at'] = pd.to_datetime(portfolio_df['updated_at'])
        
        # Safely parse JSON strings in 'tickers' column
        def safe_json_loads(x):
            try:
                # Handle potential double quotes issues
                cleaned_x = x.strip().replace('""', '"') 
                return json.loads(cleaned_x)
            except (json.JSONDecodeError, TypeError):
                return [] # Return empty list or appropriate default on error
        config_df['tickers'] = config_df['tickers'].apply(safe_json_loads)

        # Safely parse JSON strings in 'positions' column
        def safe_json_loads_dict(x):
            try:
                # Handle potential empty strings or NaN
                if pd.isna(x) or not isinstance(x, str) or not x.strip():
                    return {}
                # Handle potential double quotes issues if necessary
                cleaned_x = x.replace('""', '"') # Basic cleaning, might need more robust handling
                return json.loads(cleaned_x)
            except json.JSONDecodeError:
                 st.warning(f"Could not parse positions JSON: {x}")
                 return {} # Return empty dict on error
            except Exception as e:
                 st.warning(f"Unexpected error parsing positions JSON ({x}): {e}")
                 return {}

        portfolio_df['positions'] = portfolio_df['positions'].apply(safe_json_loads_dict)


        # Find the latest portfolio entry for each config_id
        latest_portfolio_idx = portfolio_df.loc[portfolio_df.groupby('config_id')['updated_at'].idxmax()]
        
        # Merge config data with the latest portfolio data
        # Ensure 'id' in config_df matches 'config_id' for merging
        merged_df = pd.merge(
            config_df, 
            latest_portfolio_idx, 
            left_on='id', 
            right_on='config_id', 
            how='left',
            suffixes=('_config', '_portfolio') # Add suffixes to avoid column name conflicts
        )

        # --- Transform to Streamlit App Format ---
        workflows_data = {}
        for _, row in merged_df.iterrows():
            exp_name = row['exp_name']
            config_id = row['id'] # From config_df
            positions = row['positions'] if isinstance(row['positions'], dict) else {}
            total_value = row['total_value'] if pd.notna(row['total_value']) else 0
            cash = row['cash'] if pd.notna(row['cash']) else 0

            # Calculate asset value from positions (if needed, though total_value exists)
            asset_value = sum(item.get('value', 0) for item in positions.values())
            
            # Use total_value from CSV if available, otherwise calculate
            current_total_value = total_value if pd.notna(total_value) else (cash + asset_value)


            # Generate placeholder price history for the chart (100 points for the last day)
            # In a real scenario, you'd fetch actual historical data
            now = datetime.now()
            dates = [now - timedelta(days=1) + timedelta(days=1*i/100) for i in range(101)]
            # Simple simulation based on current value - replace with real data if possible
            prices = [current_total_value * (1 + random.uniform(-0.01, 0.01)) for _ in range(100)]
            prices.insert(0, current_total_value / (1 + random.uniform(-0.01, 0.01))) # Approximate start
            percent_change = ((prices[-1] - prices[0]) / prices[0]) * 100 if prices[0] != 0 else 0


            workflows_data[exp_name] = {
                "config_id": config_id,
                "current": current_total_value,  # Use total_value from portfolio
                "change": percent_change, # Placeholder: No historical data in CSV for comparison
                "sharpe_ratio": random.uniform(0.5, 2.5),  # Placeholder
                "max_drawdown": random.uniform(2, 15), # Placeholder
                "alpha": random.uniform(-0.02, 0.08),     # Placeholder
                "win_rate": random.uniform(40, 65),    # Placeholder
                "dates": dates,                        # Placeholder dates
                "prices": prices,                       # Placeholder prices
                "positions": positions, # Use actual positions from the latest portfolio
                "cash": cash, # Add cash data
                "tickers": row['tickers'] # Keep the list of tickers from config
            }
            
        return workflows_data

    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Please ensure 'data/config_rows.csv' and 'data/portfolio_rows.csv' exist.")
        return {}
    except pd.errors.EmptyDataError as e:
        st.error(f"Data file is empty or corrupted: {e}")
        return {}
    except KeyError as e:
        st.error(f"Missing expected column in CSV file: {e}. Please check the file headers.")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return {}


# Replace the old get_workflow_data with the new loading function
# We call the new function directly here. The @st.cache_data is inside the new function.
workflows = load_and_process_data()

# Remove the old get_latest_decisions function if it exists
# @st.cache_data(ttl=300)  # 缓存5分钟
# def get_latest_decisions(portfolio_id=None):
#     # This function is no longer used
#     return [] 


# If loading failed or returned no data, show a warning
if not workflows:
    st.warning("Could not load workflow data from CSV files. Please check the files and logs.")
    # Optionally, you could fall back to example data here if desired
    # st.stop() # Stop execution if data is critical

# --- Streamlit UI Layout ---

# Top 10 Overall Workflow section (check if workflows is not empty)
if workflows:
    st.markdown("### Top 10 Overall Workflow")

    # Initialize workfow selection state
    if 'selected_workflow' not in st.session_state or st.session_state.selected_workflow not in workflows:
        st.session_state.selected_workflow = list(workflows.keys())[0] if workflows else None

    # Create workflow selection buttons
    workflow_names = list(workflows.keys())
    # Limit to top 10 or fewer if less data
    workflow_names_display = workflow_names[:10] 
    rows = (len(workflow_names_display) + 2) // 3  # Max 3 per row
    cols_per_row = min(3, len(workflow_names_display))
    selected_workflow = st.session_state.selected_workflow

    # Create integrated workflow labels
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row * cols_per_row + col_idx
            if idx < len(workflow_names_display):
                workflow_name = workflow_names_display[idx]
                # Ensure workflow_name exists in workflows before accessing
                if workflow_name in workflows:
                    workflow_data = workflows[workflow_name]
                    change = workflow_data.get("change", 0) # Use .get for safety
                    current_value = workflow_data.get("current", 0)
                    is_selected = workflow_name == selected_workflow
                    
                    # Use native Streamlit buttons
                    with cols[col_idx]:
                        # Set button title, including price and change info
                        sign = "+" if change >= 0 else ""
                        button_label = f"{workflow_name}\n${current_value:,.2f} ({sign}{change:.2f}%)"
                        
                        # Use native button
                        if st.button(
                            button_label,
                            key=f"workflow_{workflow_name}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                        ):
                            st.session_state.selected_workflow = workflow_name
                            st.rerun()  # Rerun app immediately to update UI state

    # Display chart and metrics only if a workflow is selected and exists
    if selected_workflow and selected_workflow in workflows:
        # Get the data for the selected workflow
        current_workflow_data = workflows[selected_workflow]
        workflow_change = current_workflow_data.get("change", 0) # Use .get
        
        # Chart type selection
        chart_type = st.radio(
            "Select chart type:",
            ["Line Chart"], # Candlestick requires OHLC data, which we don't have
            horizontal=True,
            key="chart_type"
        )
        
        # Main chart section
        st.subheader(f"{selected_workflow} Performance Chart (Placeholder Data)")
        
        # Initialize session state for time period if not exists
        if 'time_period' not in st.session_state:
            st.session_state.time_period = '1D' # Default to 1D as we only have placeholder 1D data
        
        # Time period buttons - Only show 1D as we only have placeholder 1D data
        if st.button(
            "1D", 
            key="period_1D",
            type="primary", # Always selected as it's the only option
            use_container_width=False # Adjust width as needed
        ):
            st.session_state.time_period = "1D"
            # No need to rerun if it's the only option

        # Get placeholder data for the chart
        detailed_data = {
            "dates": current_workflow_data.get("dates", []), # Use .get
            "prices": current_workflow_data.get("prices", []) # Use .get
        }
        
        # Create detailed chart if data exists
        fig = go.Figure()
        
        if chart_type == "Line Chart" and detailed_data["dates"] and detailed_data["prices"]:
            # Add line chart trace
            fig.add_trace(go.Scatter(
                x=detailed_data["dates"],
                y=detailed_data["prices"],
                mode='lines',
                line=dict(color='#e74c3c' if workflow_change < 0 else '#2ecc71', width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)' if workflow_change < 0 else 'rgba(46, 204, 113, 0.1)'
            ))
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Portfolio Value (Placeholder)",
                hovermode="x unified",
                margin=dict(l=0, r=0, t=0, b=0), # Minimize margins
                xaxis_showgrid=False, 
                yaxis_showgrid=True, 
                yaxis_gridcolor='rgba(200, 200, 200, 0.3)',
                plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper
            )
            # Update axis colors if needed for theme
            # fig.update_xaxes(tickfont=dict(color='grey'))
            # fig.update_yaxes(tickfont=dict(color='grey'))

        # Display the chart
        # Disable Plotly mode bar
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Performance metrics (using placeholders from loaded data)
        st.markdown("### Performance Metrics (Placeholders)")
        metric_cols = st.columns(4)
        
        # Helper to safely get metric values
        def get_metric(data, key, default_val=0.0):
             val = data.get(key, default_val)
             # Ensure the value is numeric before formatting
             return val if isinstance(val, (int, float)) else default_val

        with metric_cols[0]:
            sharpe = get_metric(current_workflow_data, 'sharpe_ratio')
            st.metric(
                label="Sharpe Ratio", 
                value=f"{sharpe:.2f}",
                # Delta is random placeholder, remove if not meaningful
                # delta=f"{random.uniform(-0.1, 0.1):.2f}" 
            )
        
        with metric_cols[1]:
            drawdown = get_metric(current_workflow_data, 'max_drawdown')
            st.metric(
                label="Max Drawdown", 
                value=f"{drawdown:.2f}%",
                # Delta is random placeholder, remove if not meaningful
                # delta=f"{random.uniform(-1, 1):.1f}%", 
                # delta_color="inverse"
            )
        
        with metric_cols[2]:
            alpha = get_metric(current_workflow_data, 'alpha')
            st.metric(
                label="Alpha", 
                value=f"{alpha:.3f}",
                 # Delta is random placeholder, remove if not meaningful
                # delta=f"{random.uniform(-0.01, 0.01):.3f}"
            )
        
        with metric_cols[3]:
            win_rate = get_metric(current_workflow_data, 'win_rate')
            st.metric(
                label="Win Rate", 
                value=f"{win_rate:.1f}%",
                 # Delta is random placeholder, remove if not meaningful
                # delta=f"{random.uniform(-2, 2):.1f}%" 
            )
        
        # --- Updated Section: Current Holdings ---
        st.markdown("### Current Holdings")
        
        # Get positions and cash from the loaded data
        positions = current_workflow_data.get("positions", {})
        cash = current_workflow_data.get("cash", 0)

        holdings_data = []
        if isinstance(positions, dict):
             for symbol, details in positions.items():
                 # Ensure details is a dictionary and has expected keys
                 if isinstance(details, dict):
                     shares = details.get("shares", 0)
                     value = details.get("value", 0)
                     # Calculate price per share if possible, handle division by zero
                     price = (value / shares) if shares else 0 
                     holdings_data.append({
                         "Symbol": symbol,
                         "Shares": shares,
                         "Price": price,
                         "Value": value
                     })
                 else:
                     st.warning(f"Unexpected format for position details of {symbol}: {details}")

        if holdings_data:
            holdings_df = pd.DataFrame(holdings_data)
            
            # Display holdings data
            st.dataframe(
                holdings_df,
                use_container_width=True,
                column_config={
                    "Shares": st.column_config.NumberColumn(
                        "Shares", format="%d"
                    ),
                    "Price": st.column_config.NumberColumn(
                        "Avg Price", format="$%.2f", help="Estimated price per share based on value/shares"
                    ),
                    "Value": st.column_config.NumberColumn(
                        "Market Value", format="$%.2f"
                    ),
                },
                hide_index=True,
            )
        else:
            st.markdown("No current holdings in this portfolio.")

        # Display Cash
        st.metric(label="Cash Balance", value=f"${cash:,.2f}")

# Add a footer or other elements if needed
# st.markdown("---")
# st.info("Data loaded from local CSV files. Metrics are placeholders.")

else:
    # This part is shown if workflows dictionary is empty after loading attempt
    st.error("Failed to load any workflow data. Cannot display the dashboard.")

# Ensure example data generation is removed or commented out if not needed as fallback
# def generate_stock_data(...): ...
# example_workflows = { ... } 
# if not workflows: workflows = example_workflows # Remove or comment this fallback
