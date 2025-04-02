import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts

st.set_page_config(layout="wide")

# --- Helper Function for Return Calculation ---
def calculate_cumulative_return(data):
    """Calculates cumulative return percentage."""
    if data.empty or data.iloc[0] == 0:
        # Avoid division by zero and handle empty data
        return pd.Series(index=data.index, dtype=float)
    # Calculate cumulative return: (current_value / first_value - 1) * 100
    cumulative_return = (data / data.iloc[0] - 1) * 100
    return cumulative_return

# Load data
try:
    config_df = pd.read_csv('data/config_rows.csv')
    # Define column names for portfolio data as it lacks a header
    portfolio_cols = ['portfolio_id', 'config_id', 'timestamp', 'cash', 'total_value', 'holdings']
    portfolio_df = pd.read_csv('data/portfolio_rows.csv', names=portfolio_cols, header=None, skiprows=1)

    # Convert timestamp to datetime objects and set as index for resampling
    portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
    # Keep a copy before setting index if needed elsewhere, although not currently
    # portfolio_df_orig_index = portfolio_df.copy()
    portfolio_df = portfolio_df.set_index('timestamp')

    # --- UI Elements ---
    st.title("Portfolio Analysis")

    # Allow user to select an experiment/workflow
    exp_name_to_id = pd.Series(config_df.id.values, index=config_df.exp_name).to_dict()
    experiment_options = ["All"] + list(exp_name_to_id.keys())
    selected_exp_name = st.selectbox("Select Experiment:", options=experiment_options)

    # Allow user to select display type (Value or Return Period)
    display_type = st.radio(
        "Select Display Type:",
        ("Daily Value", "Weekly Return", "Monthly Return", "Yearly Return"),
        horizontal=True
    )

    chart_title = ""
    echarts_series = []
    legend_data = []
    y_axis_name = ""
    y_axis_formatter = ""
    tooltip_value_formatter = None # Function for custom tooltip formatting

    # Determine Resampling Frequency and Chart Type
    resample_freq = None
    plot_returns = False
    if display_type == "Weekly Return":
        resample_freq = 'W'
        plot_returns = True
        y_axis_name = "Cumulative Return (%)"
        y_axis_formatter = '{value}%'
        tooltip_value_formatter = "function (value) { return value.toFixed(2) + '%'; }" # JS formatter
    elif display_type == "Monthly Return":
        resample_freq = 'ME' # Month End
        plot_returns = True
        y_axis_name = "Cumulative Return (%)"
        y_axis_formatter = '{value}%'
        tooltip_value_formatter = "function (value) { return value.toFixed(2) + '%'; }" # JS formatter
    elif display_type == "Yearly Return":
        resample_freq = 'YE' # Year End
        plot_returns = True
        y_axis_name = "Cumulative Return (%)"
        y_axis_formatter = '{value}%'
        tooltip_value_formatter = "function (value) { return value.toFixed(2) + '%'; }" # JS formatter
    else: # Daily Value
        y_axis_name = "Total Value ($)"
        y_axis_formatter = '${value}'
        tooltip_value_formatter = "function (value) { return '$' + value.toFixed(2); }" # JS formatter

    # --- Data Preparation Loop ---
    exp_list_to_process = []
    if selected_exp_name == "All":
        chart_title = f"{display_type} for All Experiments"
        exp_list_to_process = list(exp_name_to_id.items())
    else:
        chart_title = f"{display_type} for {selected_exp_name}"
        exp_list_to_process = [(selected_exp_name, exp_name_to_id[selected_exp_name])]

    for exp_name, config_id in exp_list_to_process:
        exp_data_full = portfolio_df[portfolio_df['config_id'] == config_id].sort_index()
        if exp_data_full.empty:
            continue # Skip if no data for this experiment

        # Select the data column to plot
        data_to_plot = exp_data_full['total_value']

        # Resample if plotting returns
        if resample_freq:
            # Use .last() to get the value at the end of the period
            data_to_plot = data_to_plot.resample(resample_freq).last()
            # Drop NaN values that can result from resampling periods with no data
            data_to_plot = data_to_plot.dropna()

        # Calculate returns if requested
        if plot_returns:
            if data_to_plot.shape[0] < 2:
                 continue # Need at least two points to calculate returns
            data_to_plot = calculate_cumulative_return(data_to_plot)
            data_to_plot = data_to_plot.dropna() # Drop NaNs from return calc if first value was 0

        if not data_to_plot.empty:
            legend_data.append(exp_name)
            # Prepare data as [timestamp_str, value] pairs for time axis
            # Ensure index is datetime before formatting
            data_pairs = [[idx.strftime('%Y-%m-%d %H:%M:%S'), val] for idx, val in data_to_plot.items()]

            series_name = exp_name
            # Optionally adjust series name if needed based on plot_returns, but experiment name is usually best
            # series_name = f"{exp_name} ({'Return' if plot_returns else 'Value'})

            echarts_series.append({
                "name": series_name,
                "data": data_pairs,
                "type": "line",
                "smooth": True,
                "symbol": "none",
                # Only show area style for single experiment daily value chart for clarity
                "areaStyle": {} if selected_exp_name != "All" and not plot_returns else None,
            })

    # --- Chart Display ---
    # Check if we have any data to plot after processing
    if echarts_series:
        # --- ECharts Configuration ---
        option = {
            "title": {
                "text": chart_title,
                "left": "center"
            },
            "tooltip": {
                "trigger": 'axis',
                "axisPointer": {
                    "type": 'cross'
                },
                # Format tooltip value based on display type
                "valueFormatter": tooltip_value_formatter
            },
            "legend": {
                "data": legend_data,
                "top": 'bottom',
                "type": 'scroll'
            },
            "grid": {
                "left": '3%',
                "right": '4%',
                "bottom": '15%',
                "containLabel": True
            },
            "xAxis": {
                "type": "time",
                "axisLabel": {
                    "formatter": '{yyyy}-{MM}-{dd}\n{HH}:{mm}:{ss}',
                    "rotate": 0
                }
            },
            "yAxis": {
                "type": "value",
                "name": y_axis_name,
                "axisLabel": {
                    "formatter": y_axis_formatter
                },
                "scale": True
            },
            "series": echarts_series,
            "dataZoom": [
                {
                    "type": "slider",
                    "xAxisIndex": 0,
                    "start": 0,
                    "end": 100,
                    "bottom": 30
                },
                {
                    "type": "inside",
                    "xAxisIndex": 0,
                    "start": 0,
                    "end": 100
                }
            ],
        }
        st_echarts(
            options=option, height="500px", # Keep height or adjust as needed
        )
    else:
        # Display warning if no data found for the selection
        if selected_exp_name == "All":
            st.warning(f"No suitable data found to display {display_type} for any experiment.")
        else:
            st.warning(f"No suitable data found to display {display_type} for experiment: {selected_exp_name}. Ensure sufficient data points exist.")

except FileNotFoundError as e:
    st.error(f"Error loading data file: {e}. Please ensure 'data/config_rows.csv' and 'data/portfolio_rows.csv' exist.")
except Exception as e:
    st.error(f"An error occurred during data processing or chart generation: {e}")
    st.exception(e) # Show detailed traceback for debugging