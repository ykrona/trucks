from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Page config
st.set_page_config(page_title="Truck Forecast Dashboard", layout="wide")

st.title("🚚 Truck Forecast - What-If Scenarios")
st.markdown("Compare different business scenarios and their impact on truck requirements")


# Add this function after your other helper functions (around line 50-100)
def extend_future_to_2026(df_train, df_future):
    """
    Extend df_future to include all 52 weeks of 2026.
    Ensures the extended data has the same structure as original df_future.
    """
    # First, check which columns were in the original df_future
    original_columns = df_future.columns.tolist()
    
    # Determine the last week in df_future
    if 'Year Hiérarchie - Year' in df_future.columns and 'Year Hiérarchie - Week' in df_future.columns:
        last_year = df_future['Year Hiérarchie - Year'].max()
        last_week = df_future['Year Hiérarchie - Week'].max()
    else:
        # If these columns don't exist in df_future, don't add them to 2026 either
        last_year = 2025
        last_week = 52
    
    print(f"Last data point: Year {last_year}, Week {last_week}")
    
    # 1. Create all 52 weeks of 2026
    weeks_2026 = []
    for week in range(1, 53):  # Weeks 1-52
        week_data = {}
        
        # Only add year/week columns if they existed in original df_future
        if 'Year Hiérarchie - Year' in original_columns:
            week_data['Year Hiérarchie - Year'] = 2026
        if 'Year Hiérarchie - Week' in original_columns:
            week_data['Year Hiérarchie - Week'] = week
        if 'weekNumbr' in original_columns:
            week_data['weekNumbr'] = week
        if 'Year Hiérarchie - Month' in original_columns:
            week_data['Year Hiérarchie - Month'] = (week - 1) // 4 + 1
        if 'Year Hiérarchie - Quarter' in original_columns:
            week_data['Year Hiérarchie - Quarter'] = (week - 1) // 13 + 1
        
        weeks_2026.append(week_data)
    
    df_2026 = pd.DataFrame(weeks_2026)
    
    # 2. Add all other columns from df_future with default values
    for col in original_columns:
        if col not in df_2026.columns:
            # Check the data type and initialize appropriately
            if df_future[col].dtype in ['float64', 'int64']:
                # For numeric columns, use the mean of the last few rows as default
                df_2026[col] = df_future[col].tail(5).mean()
            elif df_future[col].dtype == 'object':
                df_2026[col] = df_future[col].mode()[0] if len(df_future[col].mode()) > 0 else None
            else:
                df_2026[col] = None
    
    # 3. Ensure column order matches original df_future
    df_2026 = df_2026[original_columns]
    
    # 4. Combine: df_future (remaining 2025 weeks) + df_2026 (all 52 weeks of 2026)
    df_extended_future = pd.concat([df_future, df_2026], ignore_index=True)
    
    print(f"\n✅ Extended future data:")
    print(f"  Total weeks: {len(df_extended_future)}")
    if 'Year Hiérarchie - Year' in df_extended_future.columns:
        print(f"  2025 weeks: {len(df_extended_future[df_extended_future['Year Hiérarchie - Year'] == 2025])}")
        print(f"  2026 weeks: {len(df_extended_future[df_extended_future['Year Hiérarchie - Year'] == 2026])}")
    print(f"  Columns match original: {list(df_extended_future.columns) == original_columns}")
    
    return df_extended_future

# Load data function

# Modify the load_data function (around line 20-30)
@st.cache_data
def load_data():
    """Load training and future data from Excel files and extend to 2026"""
    df_train = pd.read_excel("df_train_streamlit.xlsx")
    df_future = pd.read_excel("X_future_streamlit.xlsx")
    
    # Remove 'Unnamed: 0' if it exists (Excel index column)
    if 'Unnamed: 0' in df_future.columns:
        df_future = df_future.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in df_train.columns:
        df_train = df_train.drop(columns=['Unnamed: 0'])
    
    # Extend to include 2026
    df_future_extended = extend_future_to_2026(df_train, df_future)
    
    return df_train, df_future_extended


# Add a new function to create 2026 predictions visualization
def plot_2026_predictions(all_scenarios, forecast_start_week, colors=None):
    """
    Create visualization specifically for 2026 predictions.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]
    
    if colors is None:
        colors = get_scenario_colors(all_scenarios)
    
    # Filter to only 2026 data
    # Assuming forecast data includes year info or we can calculate it
    num_forecast_weeks = len(next(iter(all_scenarios.values()))['Boisbriand'])
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{hub} - 2026 Predictions' for hub in hubs],
        vertical_spacing=0.12
    )
    
    for idx, hub in enumerate(hubs):
        row = idx + 1
        
        # For each scenario, plot 2026 weeks
        for scenario_name, scenario_data in all_scenarios.items():
            forecast = scenario_data[hub]
            
            # Assuming weeks beyond a certain point are 2026
            # You'll need to identify which indices correspond to 2026
            # This is a placeholder - adjust based on your data structure
            weeks_2026 = list(range(1, 53))
            
            # Get 2026 portion of forecast
            # This assumes the last 52 weeks of your forecast are 2026
            if len(forecast) >= 52:
                forecast_2026 = forecast[-52:]
            else:
                forecast_2026 = forecast
                weeks_2026 = list(range(1, len(forecast) + 1))
            
            if scenario_name == 'Base Forecast':
                fig.add_trace(
                    go.Scatter(
                        x=weeks_2026,
                        y=forecast_2026,
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=colors[scenario_name], width=2.5),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.2f}}<extra></extra>',
                        legendgroup=scenario_name,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=weeks_2026,
                        y=forecast_2026,
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=colors[scenario_name], width=2, dash='dash'),
                        marker=dict(size=5, symbol='square'),
                        hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.2f}}<extra></extra>',
                        legendgroup=scenario_name,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
        
        # Update axes
        fig.update_xaxes(title_text='Week Number (2026)', range=[0, 53], row=row, col=1)
        fig.update_yaxes(title_text='Number of Trucks', row=row, col=1)
    
    fig.update_layout(
        height=1200,
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0),
        title_text="2026 Truck Predictions"
    )
    
    return fig
# Load models function
@st.cache_resource
def load_models():
    """Load the trained models from pickle files"""
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    return models

def calculate_elasticity_coefficients(df_train):
    """Calculate historical elasticity relationships between business metrics and trucks"""
    elasticities = {}
    
    for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
        # Get historical data (remove NaN)
        data = df_train[[f'trucks_{hub}', 'ACTUAL SALES', 'ACTUAL TRANSACTION', 
                        'ACTUAL OUTBOUND UNIT']].dropna()
        
        # Calculate simple correlation-based elasticities
        pct_change_trucks = data[f'trucks_{hub}'].pct_change() * 100
        pct_change_sales = data['ACTUAL SALES'].pct_change() * 100
        pct_change_transactions = data['ACTUAL TRANSACTION'].pct_change() * 100
        pct_change_outbound = data['ACTUAL OUTBOUND UNIT'].pct_change() * 100
        
        # Calculate correlation
        corr_sales = pct_change_trucks.corr(pct_change_sales)
        corr_transactions = pct_change_trucks.corr(pct_change_transactions)
        corr_outbound = pct_change_trucks.corr(pct_change_outbound)
        
        # Convert correlation to elasticity estimate
        sales_elasticity = np.clip(corr_sales if not np.isnan(corr_sales) else 0.3, 0, 0.8)
        transactions_elasticity = np.clip(corr_transactions if not np.isnan(corr_transactions) else 0.5, 0, 0.8)
        outbound_elasticity = np.clip(corr_outbound if not np.isnan(corr_outbound) else 0.4, 0, 0.8)
        
        elasticities[hub] = {
            'sales': sales_elasticity,
            'transactions': transactions_elasticity,
            'outbound': outbound_elasticity
        }
    
    return elasticities

def apply_whatif_scenario(base_forecast, elasticities, hub, 
                          sales_change_pct=0, transactions_change_pct=0, 
                          outbound_change_pct=0):
    """Apply what-if scenario to base forecast using elasticity"""
    e_sales = elasticities[hub]['sales']
    e_transactions = elasticities[hub]['transactions']
    e_outbound = elasticities[hub]['outbound']
    
    truck_pct_change = (
        (sales_change_pct * e_sales) +
        (transactions_change_pct * e_transactions) +
        (outbound_change_pct * e_outbound)
    )
    
    adjustment_factor = 1 + (truck_pct_change / 100)
    adjusted_forecast = base_forecast * adjustment_factor
    
    return adjusted_forecast, truck_pct_change

def create_whatif_scenarios(df_train, X_future, models, elasticities):
    """Create what-if scenarios for all hubs"""
    # Define scenarios
    scenarios = {
        'Base Forecast': {'sales': 0, 'transactions': 0, 'outbound': 0},
        'Sales +10%': {'sales': 10, 'transactions': 0, 'outbound': 0},
        'Transactions +10%': {'sales': 0, 'transactions': 10, 'outbound': 0},
        'Sales +10% & Transactions +10%': {'sales': 10, 'transactions': 10, 'outbound': 0},
    }
    
    all_scenarios = {scenario: {} for scenario in scenarios}
    
    # Get 2024 historical data and 2025 actuals for comparison
    historical_2024 = {}
    actual_2025 = {}
    forecast_start_week = 32  # Default
    
    if 'Year Hiérarchie - Year' in df_train.columns and 'weekNumbr' in df_train.columns:
        df_2024 = df_train[df_train['Year Hiérarchie - Year'] == 2024]
        df_2025 = df_train[df_train['Year Hiérarchie - Year'] == 2025]
        
        # Determine where forecast starts (last actual week + 1)
        if len(df_2025) > 0:
            forecast_start_week = df_2025['weekNumbr'].max() + 1
        
        for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
            # Get 2024 data
            weeks_2024 = list(range(1, 53))
            trucks_2024 = []
            
            for week_num in weeks_2024:
                week_data = df_2024[df_2024['weekNumbr'] == week_num][f'trucks_{hub}']
                if len(week_data) > 0:
                    trucks_2024.append(week_data.iloc[0])
                else:
                    trucks_2024.append(0)
            
            historical_2024[hub] = np.array(trucks_2024)
            
            # Get 2025 actuals (weeks before forecast)
            weeks_2025_actual = []
            trucks_2025_actual = []
            
            for week_num in range(1, forecast_start_week):
                week_data = df_2025[df_2025['weekNumbr'] == week_num][f'trucks_{hub}']
                if len(week_data) > 0:
                    weeks_2025_actual.append(week_num)
                    trucks_2025_actual.append(week_data.iloc[0])
            
            actual_2025[hub] = {
                'weeks': weeks_2025_actual,
                'trucks': trucks_2025_actual
            }
    
    # Check if X_future already has predictions (from seasonal adjustment)
    has_predictions = all(f'pred_{hub}' in X_future.columns for hub in ["Boisbriand", "Châteauguay", "Varennes"])
    
    for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
        # If predictions already exist (from seasonal adjustment), use them
        if has_predictions:
            base_forecast = X_future[f'pred_{hub}'].values
        else:
            # Otherwise, make new predictions
            model = models[hub]
            # Prepare X_future for prediction (remove pred columns if they exist)
            X_pred = X_future.drop(columns=[c for c in X_future.columns if c.startswith('pred_')], errors='ignore')
            base_forecast = model.predict(X_pred)
        
        for scenario_name, changes in scenarios.items():
            adjusted_forecast, truck_change = apply_whatif_scenario(
                base_forecast, elasticities, hub,
                sales_change_pct=changes['sales'],
                transactions_change_pct=changes['transactions'],
                outbound_change_pct=changes['outbound']
            )
            
            all_scenarios[scenario_name][hub] = adjusted_forecast
    
    return all_scenarios, historical_2024, actual_2025, forecast_start_week


def create_dynamic_scenarios_ui():
    """
    Create a UI for users to define custom scenarios with sliders.
    Returns a dictionary of scenarios.
    """
    st.subheader("🎛️ Build Custom Scenarios")
    st.markdown("Adjust the sliders below to create custom what-if scenarios:")
    
    scenarios = {}
    
    # Base Forecast (always included)
    scenarios['Base Forecast'] = {'sales': 0, 'transactions': 0, 'outbound': 0}
    
    # Create tabs for custom scenarios
    num_custom_scenarios = st.number_input(
        "How many custom scenarios do you want to create?", 
        min_value=1, max_value=5, value=3, step=1
    )
    
    # Create columns for scenarios
    if num_custom_scenarios == 1:
        cols = [st.container()]
    elif num_custom_scenarios == 2:
        cols = st.columns(2)
    elif num_custom_scenarios == 3:
        cols = st.columns(3)
    else:
        # For 4-5 scenarios, use 2 rows
        cols_row1 = st.columns(min(num_custom_scenarios, 3))
        cols_row2 = st.columns(max(num_custom_scenarios - 3, 0)) if num_custom_scenarios > 3 else []
        cols = list(cols_row1) + list(cols_row2)
    
    for i in range(num_custom_scenarios):
        with cols[i]:
            st.markdown(f"**Scenario {i+1}**")
            
            # Scenario name
            scenario_name = st.text_input(
                f"Name", 
                value=f"Scenario {i+1}",
                key=f"scenario_name_{i}"
            )
            
            # Sales change input
            sales_change = st.number_input(
                "Sales Change (%)",
                min_value=-100.0,
                max_value=200.0,
                value=0.0,
                step=1.0,
                key=f"sales_{i}",
                help="Percentage change in sales volume (can be negative or positive)"
            )
            
            # Transactions change input
            transactions_change = st.number_input(
                "Transactions Change (%)",
                min_value=-100.0,
                max_value=200.0,
                value=0.0,
                step=1.0,
                key=f"transactions_{i}",
                help="Percentage change in transaction volume (can be negative or positive)"
            )
            
            # Outbound change input
            outbound_change = st.number_input(
                "Outbound Change (%)",
                min_value=-100.0,
                max_value=200.0,
                value=0.0,
                step=1.0,
                key=f"outbound_{i}",
                help="Percentage change in outbound units (can be negative or positive)"
            )
            
            # Add to scenarios dictionary
            scenarios[scenario_name] = {
                'sales': sales_change,
                'transactions': transactions_change,
                'outbound': outbound_change
            }
            
            # Show expected impact
            avg_impact = (sales_change + transactions_change + outbound_change) / 3
            if avg_impact > 0:
                st.info(f"📈 Avg impact: +{avg_impact:.1f}%")
            elif avg_impact < 0:
                st.warning(f"📉 Avg impact: {avg_impact:.1f}%")
            else:
                st.success("➡️ No change")
    
    return scenarios


def create_dynamic_whatif_scenarios(df_train, X_future, models, elasticities, scenarios):
    """
    Create what-if scenarios for all hubs using user-defined scenarios.
    
    Parameters:
    - df_train: training dataframe
    - X_future: future data for predictions
    - models: trained models
    - elasticities: elasticity coefficients
    - scenarios: dictionary of scenarios from create_dynamic_scenarios_ui()
    
    Returns:
    - all_scenarios: dict with forecasts for each scenario
    - historical_2024: historical data
    - actual_2025: actual 2025 data
    - forecast_start_week: week where forecast starts
    """
    
    all_scenarios = {scenario: {} for scenario in scenarios}
    
    # Get 2024 historical data and 2025 actuals for comparison
    historical_2024 = {}
    actual_2025 = {}
    forecast_start_week = 32  # Default
    
    if 'Year Hiérarchie - Year' in df_train.columns and 'weekNumbr' in df_train.columns:
        df_2024 = df_train[df_train['Year Hiérarchie - Year'] == 2024]
        df_2025 = df_train[df_train['Year Hiérarchie - Year'] == 2025]
        
        # Determine where forecast starts (last actual week + 1)
        if len(df_2025) > 0:
            forecast_start_week = df_2025['weekNumbr'].max() + 1
        
        for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
            # Get 2024 data
            weeks_2024 = list(range(1, 53))
            trucks_2024 = []
            
            for week_num in weeks_2024:
                week_data = df_2024[df_2024['weekNumbr'] == week_num][f'trucks_{hub}']
                if len(week_data) > 0:
                    trucks_2024.append(week_data.iloc[0])
                else:
                    trucks_2024.append(0)
            
            historical_2024[hub] = np.array(trucks_2024)
            
            # Get 2025 actuals (weeks before forecast)
            weeks_2025_actual = []
            trucks_2025_actual = []
            
            for week_num in range(1, forecast_start_week):
                week_data = df_2025[df_2025['weekNumbr'] == week_num][f'trucks_{hub}']
                if len(week_data) > 0:
                    weeks_2025_actual.append(week_num)
                    trucks_2025_actual.append(week_data.iloc[0])
            
            actual_2025[hub] = {
                'weeks': weeks_2025_actual,
                'trucks': trucks_2025_actual
            }
    
    # Check if X_future already has predictions (from seasonal adjustment)
    has_predictions = all(f'pred_{hub}' in X_future.columns for hub in ["Boisbriand", "Châteauguay", "Varennes"])
    
    for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
        # If predictions already exist (from seasonal adjustment), use them
        if has_predictions:
            base_forecast = X_future[f'pred_{hub}'].values
        else:
            # Otherwise, make new predictions
            model = models[hub]
            
            # Prepare X_future for prediction (remove pred columns if they exist)
            X_pred = X_future.drop(columns=[c for c in X_future.columns if c.startswith('pred_')], errors='ignore')
            
            # CRITICAL: Remove the year/quarter/month/week columns that weren't in training
            cols_to_drop = [
                'Year Hiérarchie - Year', 
                'Year Hiérarchie - Quarter', 
                'Year Hiérarchie - Month', 
                'Year Hiérarchie - Week',
                  # Add this if it exists
            ]
            X_pred = X_pred.drop(columns=[c for c in cols_to_drop if c in X_pred.columns], errors='ignore')
            
            # Also drop datetime columns
            datetime_cols = X_pred.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                X_pred = X_pred.drop(columns=datetime_cols)
            
            base_forecast = model.predict(X_pred)
        
        for scenario_name, changes in scenarios.items():
            # Import the apply_whatif_scenario function here or ensure it's available
              # Replace with actual import
            
            adjusted_forecast, truck_change = apply_whatif_scenario(
                base_forecast, elasticities, hub,
                sales_change_pct=changes['sales'],
                transactions_change_pct=changes['transactions'],
                outbound_change_pct=changes['outbound']
            )
            
            all_scenarios[scenario_name][hub] = adjusted_forecast
    
    return all_scenarios, historical_2024, actual_2025, forecast_start_week


def get_scenario_colors(scenarios):
    """
    Generate colors for each scenario dynamically.
    Base Forecast always gets gray, others get distinct colors.
    """
    # Predefined color palette
    color_palette = [
        '#3498db',  # Blue
        '#e74c3c',  # Red
        '#9b59b6',  # Purple
        '#f39c12',  # Orange
        '#1abc9c',  # Teal
        '#e67e22',  # Dark Orange
        '#2ecc71',  # Green
        '#34495e',  # Dark Gray
    ]
    
    colors = {}
    colors['Base Forecast'] = '#95a5a6'  # Gray for base
    
    color_idx = 0
    for scenario_name in scenarios:
        if scenario_name != 'Base Forecast':
            colors[scenario_name] = color_palette[color_idx % len(color_palette)]
            color_idx += 1
    
    return colors

from scipy import stats
import numpy as np

def calculate_confidence_intervals(df_train, models, confidence_level=0.95):
    """
    Calculate confidence intervals based on historical prediction errors.
    Returns margin of error for each hub.
    """
    intervals = {}
    
    # Prepare training data (same preprocessing as your main code)
    cols_to_drop = [
        "trucks_Boisbriand", "trucks_Châteauguay", "trucks_Varennes",
        'Year Hiérarchie - Year', 'Year Hiérarchie - Quarter', 
        'Year Hiérarchie - Month', 'Year Hiérarchie - Week',
        "TREND 15% GROWTH", "TRUCK FLEET", "Somme de Truck Forecast", 
        "FORECAST +15%", "TRUCK RENTAL V1", "FORECAST SALES GROWTH 15%", 
        "TREND FORECAST SALES", "FORECAST SALES BUDGET", 
        "FORECAST TRANSACTION GROWTH 15%", "TREND FINANCE TRANSACTION", 
        "Somme de BUDGET TRANSACTION", "ACTUAL TRUCK USED", 
        "ACTUAL SALES", "ACTUAL TRANSACTION", 
        "ACTUAL OUTBOUND UNIT", "ACTUAL INBOUND UNIT"
    ]
    
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    
    # Remove datetime columns
    datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        X_train = X_train.drop(columns=datetime_cols)
    
    # Remove any prediction columns
    pred_cols = [c for c in X_train.columns if 'pred_' in c or 'MAPE' in c]
    if pred_cols:
        X_train = X_train.drop(columns=pred_cols)
    
    # Z-score for confidence level (1.96 for 95%)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
        # Get actual values (drop NaN)
        y_actual = df_train[f'trucks_{hub}'].dropna()
        
        # Align X_train with y_actual
        X_train_aligned = X_train.loc[y_actual.index]
        
        # Get predictions on training data
        y_pred = models[hub].predict(X_train_aligned)
        
        # Calculate residuals
        residuals = y_actual.values - y_pred
        
        # Calculate standard deviation of residuals
        std_error = np.std(residuals)
        
        # Calculate margin of error
        margin = z_score * std_error
        
        intervals[hub] = {
            'std_error': std_error,
            'margin': margin,
            'mean_absolute_error': np.mean(np.abs(residuals))
        }
    
    return intervals


def add_confidence_bands_to_plot(fig, forecast_week_indices, forecast, intervals, hub, 
                                  color, row, idx, scenario_name='Base Forecast'):
    """
    Add confidence interval bands to an existing Plotly figure.
    
    Parameters:
    - fig: Plotly figure object
    - forecast_week_indices: list of week numbers for x-axis
    - forecast: array of forecasted values
    - intervals: dict with margin of error for each hub
    - hub: hub name
    - color: color for the confidence band
    - row: subplot row number
    - idx: hub index (for legend control)
    - scenario_name: name of the scenario
    """
    upper_bound = forecast + intervals[hub]['margin']
    lower_bound = np.maximum(forecast - intervals[hub]['margin'], 0)  # Can't have negative trucks
    
    # Create the confidence band using filled area
    fig.add_trace(
        go.Scatter(
            x=forecast_week_indices + forecast_week_indices[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill='toself',
            fillcolor=color,
            opacity=0.15,
            line=dict(width=0),
            name=f'95% Confidence Interval',
            hovertemplate=f'<b>{scenario_name} 95% CI</b><br>Week: %{{x}}<br>Range: {intervals[hub]["margin"]:.1f} trucks<extra></extra>',
            legendgroup=f'{scenario_name}_ci',
            showlegend=(idx == 0 and scenario_name == 'Base Forecast')  # Only show once
        ),
        row=row, col=1
    )
    
    return fig


# Modified plot_scenarios function excerpt showing where to add CI:
def plot_scenarios_with_ci(all_scenarios, intervals, historical_2024=None, 
                           actual_2025=None, forecast_start_week=32):
    """
    Modified version of plot_scenarios that includes confidence intervals.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]
    colors = {
        'Base Forecast': '#95a5a6',
        'Sales +10%': '#3498db',
        'Transactions +10%': '#e74c3c',
        'Sales +10% & Transactions +10%': '#9b59b6'
    }
    
    num_forecast_weeks = len(next(iter(all_scenarios.values()))['Boisbriand'])
    forecast_week_indices = list(range(forecast_start_week, forecast_start_week + num_forecast_weeks))
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{hub} - What-If Scenarios' for hub in hubs],
        vertical_spacing=0.12
    )
    
    for idx, hub in enumerate(hubs):
        row = idx + 1
        
        # Add 2024 historical data if available
        if historical_2024 and hub in historical_2024:
            weeks_2024 = list(range(1, 53))
            trucks_2024 = historical_2024[hub]
            
            fig.add_trace(
                go.Scatter(
                    x=weeks_2024,
                    y=trucks_2024,
                    mode='lines+markers',
                    name='2024 Actual',
                    line=dict(color='#34495e', width=2, dash='dot'),
                    marker=dict(size=4),
                    opacity=0.6,
                    hovertemplate='<b>2024 Actual</b><br>Week: %{x}<br>Trucks: %{y:.2f}<extra></extra>',
                    legendgroup='2024_actual',
                    showlegend=(idx == 0)
                ),
                row=row, col=1
            )
        
        # Add 2025 actual data if available
        if actual_2025 and hub in actual_2025:
            weeks_2025_actual = actual_2025[hub]['weeks']
            trucks_2025_actual = actual_2025[hub]['trucks']
            
            if weeks_2025_actual:
                fig.add_trace(
                    go.Scatter(
                        x=weeks_2025_actual,
                        y=trucks_2025_actual,
                        mode='lines+markers',
                        name='2025 Actual',
                        line=dict(color='#2E86AB', width=2.5),
                        marker=dict(size=5),
                        hovertemplate='<b>2025 Actual</b><br>Week: %{x}<br>Trucks: %{y:.2f}<extra></extra>',
                        legendgroup='2025_actual',
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
        
        # Add scenario forecasts with confidence intervals
        for scenario_name, scenario_data in all_scenarios.items():
            forecast = scenario_data[hub]
            
            # Add confidence interval FIRST (so it appears behind the line)
            if scenario_name == 'Base Forecast' and intervals is not None:
                fig = add_confidence_bands_to_plot(
                    fig, forecast_week_indices, forecast, intervals, hub,
                    colors[scenario_name], row, idx, scenario_name
                )
            
            # Then add the forecast line
            if scenario_name == 'Base Forecast':
                fig.add_trace(
                    go.Scatter(
                        x=forecast_week_indices,
                        y=forecast,
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=colors[scenario_name], width=2.5),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.2f}}<extra></extra>',
                        legendgroup=scenario_name,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_week_indices,
                        y=forecast,
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=colors[scenario_name], width=2, dash='dash'),
                        marker=dict(size=5, symbol='square'),
                        hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.2f}}<extra></extra>',
                        legendgroup=scenario_name,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
        
        # Add vertical line at forecast start
        fig.add_vline(
            x=forecast_start_week - 0.5,
            line_dash="dot",
            line_color="red",
            opacity=0.6,
            row=row, col=1,
            annotation_text=f"Forecast Start (Week {forecast_start_week})" if idx == 0 else "",
            annotation_position="top"
        )
        
        # Add shaded forecast region
        fig.add_vrect(
            x0=forecast_start_week - 0.5, x1=52,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            row=row, col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text='Week Number', range=[0, 53], row=row, col=1)
        fig.update_yaxes(title_text='Number of Trucks', row=row, col=1)
    
    fig.update_layout(
        height=1200,
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0)
    )
    
    return fig





def plot_scenarios(all_scenarios, historical_2024=None, actual_2025=None, forecast_start_week=32, intervals=None, colors=None):
    """Create the scenario visualization with Plotly for interactivity"""
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]

# Use provided colors or default
    if colors is None:
        colors = {
            'Base Forecast': '#95a5a6',
            'Sales +10%': '#3498db',
            'Transactions +10%': '#e74c3c',
            'Sales +10% & Transactions +10%': '#9b59b6'
        }

    # colors = {
    #     'Base Forecast': '#95a5a6',
    #     'Sales +10%': '#3498db',
    #     'Transactions +10%': '#e74c3c',
    #     'Sales +10% & Transactions +10%': '#9b59b6'
    # }
    
    num_forecast_weeks = len(next(iter(all_scenarios.values()))['Boisbriand'])
    
    # Forecast weeks start from forecast_start_week
    forecast_week_indices = list(range(forecast_start_week, forecast_start_week + num_forecast_weeks))
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{hub} - What-If Scenarios' for hub in hubs],
        vertical_spacing=0.12
    )
    
    for idx, hub in enumerate(hubs):
        row = idx + 1
        
        # Add 2024 historical data if available
        if historical_2024 and hub in historical_2024:
            weeks_2024 = list(range(1, 53))
            trucks_2024 = historical_2024[hub]
            
            fig.add_trace(
                go.Scatter(
                    x=weeks_2024,
                    y=trucks_2024,
                    mode='lines+markers',
                    name='2024 Actual',
                    line=dict(color='#34495e', width=2, dash='dot'),
                    marker=dict(size=4),
                    opacity=0.6,
                    hovertemplate='<b>2024 Actual</b><br>Week: %{x}<br>Trucks: %{y:.2f}<extra></extra>',
                    legendgroup='2024_actual',
                    showlegend=(idx == 0)
                ),
                row=row, col=1
            )
        
        # Add 2025 actual data if available
        if actual_2025 and hub in actual_2025:
            weeks_2025_actual = actual_2025[hub]['weeks']
            trucks_2025_actual = actual_2025[hub]['trucks']
            
            if weeks_2025_actual:
                fig.add_trace(
                    go.Scatter(
                        x=weeks_2025_actual,
                        y=trucks_2025_actual,
                        mode='lines+markers',
                        name='2025 Actual',
                        line=dict(color='#2E86AB', width=2.5),
                        marker=dict(size=5),
                        hovertemplate='<b>2025 Actual</b><br>Week: %{x}<br>Trucks: %{y:.2f}<extra></extra>',
                        legendgroup='2025_actual',
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
        
        # Add scenario forecasts (starting from forecast_start_week)
        for scenario_name, scenario_data in all_scenarios.items():
            forecast = scenario_data[hub]

            if scenario_name == 'Base Forecast' and intervals is not None:
                fig = add_confidence_bands_to_plot(
                fig, forecast_week_indices, forecast, intervals, hub,
                colors[scenario_name], row, idx, scenario_name
                )
            
            if scenario_name == 'Base Forecast':
                fig.add_trace(
                    go.Scatter(
                        x=forecast_week_indices,
                        y=forecast,
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=colors[scenario_name], width=2.5),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.2f}}<extra></extra>',
                        legendgroup=scenario_name,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_week_indices,
                        y=forecast,
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=colors[scenario_name], width=2, dash='dash'),
                        marker=dict(size=5, symbol='square'),
                        hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.2f}}<extra></extra>',
                        legendgroup=scenario_name,
                        showlegend=(idx == 0)
                    ),
                    row=row, col=1
                )
        
        # Add vertical line at forecast start
        fig.add_vline(
            x=forecast_start_week - 0.5,
            line_dash="dot",
            line_color="red",
            opacity=0.6,
            row=row, col=1,
            annotation_text=f"Forecast Start (Week {forecast_start_week})" if idx == 0 else "",
            annotation_position="top"
        )
        
        # Add shaded forecast region
        fig.add_vrect(
            x0=forecast_start_week - 0.5, x1=52,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            row=row, col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text='Week Number', range=[0, 53], row=row, col=1)
        fig.update_yaxes(title_text='Number of Trucks', row=row, col=1)
    
    fig.update_layout(
        height=1200,
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0)
    )
    
    return fig


def plot_scenarios_combined(all_scenarios, intervals, historical_2024=None, 
                           actual_2025=None, forecast_start_week=32, colors=None):
    """
    Create a combined visualization showing total trucks across all hubs
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]

    if colors is None:
        colors = {
            'Base Forecast': '#95a5a6',
            'Sales +10%': '#3498db',
            'Transactions +10%': '#e74c3c',
            'Sales +10% & Transactions +10%': '#9b59b6'
        }

    # colors = {
    #     'Base Forecast': '#95a5a6',
    #     'Sales +10%': '#3498db',
    #     'Transactions +10%': '#e74c3c',
    #     'Sales +10% & Transactions +10%': '#9b59b6'
    # }
    
    num_forecast_weeks = len(next(iter(all_scenarios.values()))['Boisbriand'])
    forecast_week_indices = list(range(forecast_start_week, forecast_start_week + num_forecast_weeks))
    
    # Create single plot
    fig = go.Figure()
    
    # Aggregate 2024 historical data across all hubs
    if historical_2024:
        weeks_2024 = list(range(1, 53))
        total_trucks_2024 = np.zeros(52)
        
        for hub in hubs:
            if hub in historical_2024:
                total_trucks_2024 += historical_2024[hub]
        
        fig.add_trace(
            go.Scatter(
                x=weeks_2024,
                y=total_trucks_2024,
                mode='lines+markers',
                name='2024 Actual (All Hubs)',
                line=dict(color='#34495e', width=2.5, dash='dot'),
                marker=dict(size=5),
                opacity=0.7,
                hovertemplate='<b>2024 Actual Total</b><br>Week: %{x}<br>Trucks: %{y:.1f}<extra></extra>'
            )
        )
        
        fig.update_layout(
                height=1200,
                hovermode='closest',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0)
            )
    # Aggregate 2025 actual data across all hubs
    if actual_2025:
        # Find the common weeks that exist across all hubs
        all_weeks = []
        for hub in hubs:
            if hub in actual_2025:
                all_weeks.extend(actual_2025[hub]['weeks'])
        
        if all_weeks:
            unique_weeks = sorted(set(all_weeks))
            total_trucks_2025_actual = []
            
            for week in unique_weeks:
                week_total = 0
                for hub in hubs:
                    if hub in actual_2025:
                        if week in actual_2025[hub]['weeks']:
                            idx = actual_2025[hub]['weeks'].index(week)
                            week_total += actual_2025[hub]['trucks'][idx]
                total_trucks_2025_actual.append(week_total)
            
            fig.add_trace(
                go.Scatter(
                    x=unique_weeks,
                    y=total_trucks_2025_actual,
                    mode='lines+markers',
                    name='2025 Actual (All Hubs)',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>2025 Actual Total</b><br>Week: %{x}<br>Trucks: %{y:.1f}<extra></extra>'
                )
            )
    
    # Aggregate scenario forecasts across all hubs
    for scenario_name, scenario_data in all_scenarios.items():
        # Sum forecasts across all hubs
        total_forecast = np.zeros(num_forecast_weeks)
        for hub in hubs:
            total_forecast += scenario_data[hub]
        
        # Calculate combined confidence interval for Base Forecast
        if scenario_name == 'Base Forecast' and intervals is not None:
            # Sum the margins (conservative approach - assumes independence)
            total_margin = np.sqrt(sum([intervals[hub]['margin']**2 for hub in hubs]))
            
            upper_bound = total_forecast + total_margin
            lower_bound = np.maximum(total_forecast - total_margin, 0)
            
            # Add confidence band
            fig.add_trace(
                go.Scatter(
                    x=forecast_week_indices + forecast_week_indices[::-1],
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor=colors[scenario_name],
                    opacity=0.15,
                    line=dict(width=0),
                    name='95% Confidence Interval',
                    hovertemplate=f'<b>95% CI</b><br>Week: %{{x}}<br>Margin: ±{total_margin:.1f} trucks<extra></extra>',
                    showlegend=True
                )
            )
        
        # Add the forecast line
        if scenario_name == 'Base Forecast':
            fig.add_trace(
                go.Scatter(
                    x=forecast_week_indices,
                    y=total_forecast,
                    mode='lines+markers',
                    name=scenario_name,
                    line=dict(color=colors[scenario_name], width=3),
                    marker=dict(size=7),
                    hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.1f}}<extra></extra>'
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=forecast_week_indices,
                    y=total_forecast,
                    mode='lines+markers',
                    name=scenario_name,
                    line=dict(color=colors[scenario_name], width=2.5, dash='dash'),
                    marker=dict(size=6, symbol='square'),
                    hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Trucks: %{{y:.1f}}<extra></extra>'
                )
            )
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=forecast_start_week - 0.5,
        line_dash="dot",
        line_color="red",
        opacity=0.6,
        annotation_text=f"Forecast Start (Week {forecast_start_week})",
        annotation_position="top"
    )
    
    # Add shaded forecast region
    fig.add_vrect(
        x0=forecast_start_week - 0.5, x1=52,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Total Trucks Across All Hubs - What-If Scenarios',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title='Week Number',
        yaxis_title='Total Number of Trucks (All Hubs)',
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        xaxis=dict(range=[0, 53]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def apply_seasonal_adjustment_to_forecast(df_train, df_future, models):
    """
    Calculate historical seasonal drop for weeks 47-48 and apply to forecast.
    """
    adjustment_factors = {}
    
    # First, make base predictions if not already done
    X_future = df_future.copy()
    
    # Remove columns that shouldn't be used for prediction
    cols_to_drop = [
        "trucks_Boisbriand", "trucks_Châteauguay", "trucks_Varennes",
        'Year Hiérarchie - Year', 'Year Hiérarchie - Quarter', 
        'Year Hiérarchie - Month', 'Year Hiérarchie - Week',
        "TREND 15% GROWTH", "TRUCK FLEET", "Somme de Truck Forecast", 
        "FORECAST +15%", "TRUCK RENTAL V1", "FORECAST SALES GROWTH 15%", 
        "TREND FORECAST SALES", "FORECAST SALES BUDGET", 
        "FORECAST TRANSACTION GROWTH 15%", "TREND FINANCE TRANSACTION", 
        "Somme de BUDGET TRANSACTION", "ACTUAL TRUCK USED", 
        "ACTUAL SALES", "ACTUAL TRANSACTION", 
        "ACTUAL OUTBOUND UNIT", "ACTUAL INBOUND UNIT"
    ]
    
    X_future = X_future.drop(columns=[c for c in cols_to_drop if c in X_future.columns], errors='ignore')
    
    # Remove datetime and prediction columns
    datetime_cols = X_future.select_dtypes(include=['datetime64']).columns
    pred_cols = [c for c in X_future.columns if 'pred_' in c or 'MAPE' in c]
    drop_cols = list(datetime_cols) + pred_cols
    if drop_cols:
        X_future = X_future.drop(columns=[c for c in drop_cols if c in X_future.columns])
    
    # Make a copy of df_future for adjustments
    df_future_adjusted = df_future.copy()
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]
    
    for hub in hubs:
        # Calculate historical seasonal pattern
        if 'weekNumbr' in df_train.columns:
            week_47_48_data = df_train[df_train['weekNumbr'].isin([47, 48])][f'trucks_{hub}']
            surrounding_weeks_data = df_train[df_train['weekNumbr'].isin([45, 46, 49, 50])][f'trucks_{hub}']
            
            if len(week_47_48_data) > 0 and len(surrounding_weeks_data) > 0:
                avg_week_47_48 = week_47_48_data.mean()
                avg_surrounding = surrounding_weeks_data.mean()
                adjustment_factor = avg_week_47_48 / avg_surrounding if avg_surrounding > 0 else 1.0
                adjustment_factors[hub] = adjustment_factor
            else:
                adjustment_factors[hub] = 1.0
        else:
            adjustment_factors[hub] = 1.0
        
        # Make predictions if not already done
        if f'pred_{hub}' not in df_future_adjusted.columns:
            model = models[hub]
            predictions = model.predict(X_future)
            df_future_adjusted[f'pred_{hub}'] = predictions
        
        # Apply adjustment to weeks 47-48 in forecast
        if 'weekNumbr' in df_future_adjusted.columns:
            mask_47_48 = df_future_adjusted['weekNumbr'].isin([47, 48])
            indices_47_48 = df_future_adjusted[mask_47_48].index
            
            if len(indices_47_48) > 0:
                df_future_adjusted.loc[indices_47_48, f'pred_{hub}'] *= adjustment_factors[hub]
    
    return df_future_adjusted, adjustment_factors

def create_yoy_comparison(df_train, df_future, models):
    """
    Create year-over-year comparison: 2024 actuals vs 2025 (actuals + forecast).
    """
    # Prepare future predictions
    X_future = df_future.copy()
    
    # Remove columns that shouldn't be used for prediction
    cols_to_drop = [
        "trucks_Boisbriand", "trucks_Châteauguay", "trucks_Varennes",
        'Year Hiérarchie - Year', 'Year Hiérarchie - Quarter', 
        'Year Hiérarchie - Month', 'Year Hiérarchie - Week',
        "TREND 15% GROWTH", "TRUCK FLEET", "Somme de Truck Forecast", 
        "FORECAST +15%", "TRUCK RENTAL V1", "FORECAST SALES GROWTH 15%", 
        "TREND FORECAST SALES", "FORECAST SALES BUDGET", 
        "FORECAST TRANSACTION GROWTH 15%", "TREND FINANCE TRANSACTION", 
        "Somme de BUDGET TRANSACTION", "ACTUAL TRUCK USED", 
        "ACTUAL SALES", "ACTUAL TRANSACTION", 
        "ACTUAL OUTBOUND UNIT", "ACTUAL INBOUND UNIT"
    ]
    
    X_future = X_future.drop(columns=[c for c in cols_to_drop if c in X_future.columns], errors='ignore')
    
    # Remove datetime and prediction columns
    datetime_cols = X_future.select_dtypes(include=['datetime64']).columns
    pred_cols = [c for c in X_future.columns if 'pred_' in c or 'MAPE' in c]
    drop_cols = list(datetime_cols) + pred_cols
    if drop_cols:
        X_future = X_future.drop(columns=[c for c in drop_cols if c in X_future.columns])
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]
    colors_2024 = ['#34495e', '#7f8c8d', '#95a5a6']
    colors_2025 = ['#2E86AB', '#A23B72', '#F18F01']
    
    comparison_data = {}
    
    # Determine split point: find where 2024 ends and 2025 begins
    if 'Year Hiérarchie - Year' in df_train.columns:
        df_2024 = df_train[df_train['Year Hiérarchie - Year'] == 2024]
        df_2025_actual = df_train[df_train['Year Hiérarchie - Year'] == 2025]
    else:
        split_idx = len(df_train) - 31
        df_2024 = df_train.iloc[:split_idx]
        df_2025_actual = df_train.iloc[split_idx:]
    
    # Create Plotly subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'{hub}: 2024 vs 2025 Year-over-Year Comparison' for hub in hubs],
        vertical_spacing=0.1
    )
    
    for idx, hub in enumerate(hubs):
        row = idx + 1
        
        # 2024 DATA: Get weeks from df_2024
        weeks_2024 = list(range(1, 53))
        trucks_2024 = [0] * 52
        
        if 'weekNumbr' in df_2024.columns:
            for i, week_num in enumerate(weeks_2024):
                week_data = df_2024[df_2024['weekNumbr'] == week_num][f'trucks_{hub}']
                if len(week_data) > 0:
                    trucks_2024[i] = week_data.iloc[0]
        
        # 2025 DATA: Actuals + Forecast
        weeks_2025 = list(range(1, 53))
        trucks_2025 = [0] * 52
        
        # Get 2025 actuals
        if 'weekNumbr' in df_2025_actual.columns:
            for i, week_num in enumerate(weeks_2025):
                week_data = df_2025_actual[df_2025_actual['weekNumbr'] == week_num][f'trucks_{hub}']
                if len(week_data) > 0:
                    trucks_2025[i] = week_data.iloc[0]
        
        # Add 2025 forecast
        if 'weekNumbr' in df_future.columns:
            for i, week_num in enumerate(weeks_2025):
                week_data = df_future[df_future['weekNumbr'] == week_num]
                if len(week_data) > 0:
                    if f'pred_{hub}' in df_future.columns:
                        trucks_2025[i] = week_data.iloc[0][f'pred_{hub}']
                    else:
                        row_idx = week_data.index[0]
                        model = models[hub]
                        future_row_idx = df_future.index.get_loc(row_idx)
                        prediction = model.predict(X_future.iloc[future_row_idx:future_row_idx+1])
                        trucks_2025[i] = prediction[0]
        
        comparison_data[hub] = {
            'weeks': weeks_2024,
            '2024': trucks_2024,
            '2025': trucks_2025
        }
        
        # Plot 2024 actual
        fig.add_trace(
            go.Scatter(
                x=weeks_2024,
                y=trucks_2024,
                mode='lines+markers',
                name='2024 Actual',
                line=dict(color=colors_2024[idx], width=2.5),
                marker=dict(size=5),
                hovertemplate='<b>2024 Actual</b><br>Week: %{x}<br>Trucks: %{y:.1f}<extra></extra>',
                legendgroup='2024',
                showlegend=(idx == 0)
            ),
            row=row, col=1
        )
        
        # Split 2025 into actual and forecast
        last_actual_week = df_2025_actual['weekNumbr'].max() if 'weekNumbr' in df_2025_actual.columns else 31
        
        weeks_2025_actual = [w for w in weeks_2025 if w <= last_actual_week]
        trucks_2025_actual = [trucks_2025[i] for i in range(len(weeks_2025)) if weeks_2025[i] <= last_actual_week]
        
        weeks_2025_forecast = [w for w in weeks_2025 if w > last_actual_week]
        trucks_2025_forecast = [trucks_2025[i] for i in range(len(weeks_2025)) if weeks_2025[i] > last_actual_week]
        
        # Plot 2025 actual
        if trucks_2025_actual:
            fig.add_trace(
                go.Scatter(
                    x=weeks_2025_actual,
                    y=trucks_2025_actual,
                    mode='lines+markers',
                    name='2025 Actual',
                    line=dict(color=colors_2025[idx], width=2.5),
                    marker=dict(size=5),
                    hovertemplate='<b>2025 Actual</b><br>Week: %{x}<br>Trucks: %{y:.1f}<extra></extra>',
                    legendgroup='2025_actual',
                    showlegend=(idx == 0)
                ),
                row=row, col=1
            )
        
        # Plot 2025 forecast
        if trucks_2025_forecast:
            fig.add_trace(
                go.Scatter(
                    x=weeks_2025_forecast,
                    y=trucks_2025_forecast,
                    mode='lines+markers',
                    name='2025 Forecast',
                    line=dict(color=colors_2025[idx], width=2.5, dash='dash'),
                    marker=dict(size=6, symbol='square'),
                    hovertemplate='<b>2025 Forecast</b><br>Week: %{x}<br>Trucks: %{y:.1f}<extra></extra>',
                    legendgroup='2025_forecast',
                    showlegend=(idx == 0)
                ),
                row=row, col=1
            )
        
        # Add vertical line at forecast start
        fig.add_vline(
            x=last_actual_week,
            line_dash="dot",
            line_color="red",
            opacity=0.6,
            row=row, col=1,
            annotation_text=f"Forecast Start (Week {last_actual_week+1})" if idx == 0 else "",
            annotation_position="top"
        )
        
        # Add shaded forecast region
        fig.add_vrect(
            x0=last_actual_week, x1=52,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            row=row, col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text='Week Number', range=[0, 53], row=row, col=1)
        fig.update_yaxes(title_text='Number of Trucks', row=row, col=1)
        
        # Add stats annotation
        avg_2024 = np.mean([t for t in trucks_2024 if t > 0])
        avg_2025_actual = np.mean([t for t in trucks_2025_actual if t > 0]) if trucks_2025_actual else 0
        avg_2025_forecast = np.mean([t for t in trucks_2025_forecast if t > 0]) if trucks_2025_forecast else 0
        avg_2025_total = np.mean([t for t in trucks_2025 if t > 0])
        yoy_change = ((avg_2025_total - avg_2024) / avg_2024) * 100 if avg_2024 > 0 else 0
        
        stats_text = (f'2024 Avg: {avg_2024:.1f} trucks<br>'
                     f'2025 Actual Avg: {avg_2025_actual:.1f} trucks<br>'
                     f'2025 Forecast Avg: {avg_2025_forecast:.1f} trucks<br>'
                     f'2025 Total Avg: {avg_2025_total:.1f} trucks<br>'
                     f'YoY Change: {yoy_change:+.1f}%')
        
        # Correctly format the xref/yref for subplots
        xref = 'x domain' if row == 1 else f'x{row} domain'
        yref = 'y domain' if row == 1 else f'y{row} domain'
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref=xref, yref=yref,
            text=stats_text,
            showarrow=False,
            bgcolor='wheat',
            opacity=0.7,
            borderpad=10,
            align='left',
            xanchor='left',
            yanchor='top'
        )
    
    fig.update_layout(
        height=1400,
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0)
    )
    
    # Create comparison DataFrame
    export_data = {'Week': list(range(1, 53))}
    
    for hub in hubs:
        export_data[f'{hub}_2024'] = comparison_data[hub]['2024']
        export_data[f'{hub}_2025'] = comparison_data[hub]['2025']
    
    df_comparison = pd.DataFrame(export_data)
    
    # Calculate differences
    for hub in hubs:
        df_comparison[f'{hub}_Difference'] = (
            df_comparison[f'{hub}_2025'] - df_comparison[f'{hub}_2024']
        )
        df_comparison[f'{hub}_Change_%'] = (
            (df_comparison[f'{hub}_Difference'] / 
             df_comparison[f'{hub}_2024'].replace(0, np.nan)) * 100
        )
    
    return fig, df_comparison



import numpy as np
import pandas as pd

def calculate_rental_costs(all_scenarios, forecast_start_week, fleet_capacity=48, cost_per_week=4500):
    """
    Calculate truck rental costs for scenarios where total trucks exceed fleet capacity.
    
    Parameters:
    - all_scenarios: dict of scenarios with forecasts for each hub
    - forecast_start_week: week number where forecast starts
    - fleet_capacity: total available trucks (default 48)
    - cost_per_week: cost per additional truck per week (default $4500)
    
    Returns:
    - DataFrame with rental cost analysis for each scenario
    """
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]
    results = []
    
    for scenario_name, scenario_data in all_scenarios.items():
        # Calculate total trucks per week across all hubs
        num_weeks = len(scenario_data[hubs[0]])
        total_trucks_per_week = np.zeros(num_weeks)
        
        for hub in hubs:
            total_trucks_per_week += scenario_data[hub]
        
        # Calculate excess trucks (above capacity)
        excess_trucks_per_week = np.maximum(total_trucks_per_week - fleet_capacity, 0)
        
        # Count weeks where we exceed capacity
        weeks_over_capacity = np.sum(excess_trucks_per_week > 0)
        
        # Calculate total excess truck-weeks (sum of all excess)
        total_excess_truck_weeks = np.sum(excess_trucks_per_week)
        
        # Calculate total cost
        total_rental_cost = total_excess_truck_weeks * cost_per_week
        
        # Calculate average excess when over capacity
        avg_excess_when_over = np.mean(excess_trucks_per_week[excess_trucks_per_week > 0]) if weeks_over_capacity > 0 else 0
        
        # Find peak excess
        max_excess = np.max(excess_trucks_per_week)
        week_of_max_excess = forecast_start_week + np.argmax(excess_trucks_per_week) if max_excess > 0 else None
        
        results.append({
            'Scenario': scenario_name,
            'Weeks Over Capacity': int(weeks_over_capacity),
            'Total Excess Truck-Weeks': f"{total_excess_truck_weeks:.1f}",
            'Average Excess (when over)': f"{avg_excess_when_over:.1f}",
            'Peak Excess Trucks': f"{max_excess:.1f}",
            'Week of Peak': int(week_of_max_excess) if week_of_max_excess else '-',
            'Total Rental Cost': f"${total_rental_cost:,.0f}",
            'Cost (numeric)': total_rental_cost  # For sorting
        })
    
    df_costs = pd.DataFrame(results)
    return df_costs


def calculate_annual_rental_costs(all_scenarios, historical_2024, actual_2025, forecast_start_week, 
                                   fleet_capacity=48, cost_per_week=4500):
    """
    Calculate rental costs for the full year (2025 actuals + forecast).
    
    Parameters:
    - all_scenarios: dict of scenarios with forecasts for each hub
    - historical_2024: dict with 2024 historical data
    - actual_2025: dict with 2025 actual data
    - forecast_start_week: week number where forecast starts
    - fleet_capacity: total available trucks (default 48)
    - cost_per_week: cost per additional truck per week (default $4500)
    
    Returns:
    - DataFrame with full year rental cost analysis
    """
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]
    results = []
    
    # First, compile full year data (actuals + forecast)
    for scenario_name, scenario_data in all_scenarios.items():
        # Initialize full year array (52 weeks)
        total_trucks_full_year = np.zeros(52)
        
        # Add 2025 actuals (weeks before forecast_start_week)
        if actual_2025:
            for week_idx in range(1, forecast_start_week):
                week_total = 0
                for hub in hubs:
                    if hub in actual_2025 and week_idx in actual_2025[hub]['weeks']:
                        idx = actual_2025[hub]['weeks'].index(week_idx)
                        week_total += actual_2025[hub]['trucks'][idx]
                total_trucks_full_year[week_idx - 1] = week_total
        
        # Add forecast data (from forecast_start_week onwards)
        forecast_weeks = len(scenario_data[hubs[0]])
        for i in range(forecast_weeks):
            week_idx = forecast_start_week + i - 1  # Convert to 0-indexed
            if week_idx < 52:
                week_total = sum([scenario_data[hub][i] for hub in hubs])
                total_trucks_full_year[week_idx] = week_total
        
        # Calculate excess trucks (above capacity)
        excess_trucks_per_week = np.maximum(total_trucks_full_year - fleet_capacity, 0)
        
        # Count weeks where we exceed capacity
        weeks_over_capacity = np.sum(excess_trucks_per_week > 0)
        
        # Calculate total excess truck-weeks
        total_excess_truck_weeks = np.sum(excess_trucks_per_week)
        
        # Calculate total cost
        total_rental_cost = total_excess_truck_weeks * cost_per_week
        
        # Calculate average excess when over capacity
        avg_excess_when_over = np.mean(excess_trucks_per_week[excess_trucks_per_week > 0]) if weeks_over_capacity > 0 else 0
        
        # Find peak excess
        max_excess = np.max(excess_trucks_per_week)
        week_of_max_excess = np.argmax(excess_trucks_per_week) + 1 if max_excess > 0 else None
        
        # Calculate actual vs forecast breakdown
        actual_weeks = forecast_start_week - 1
        actual_excess = np.sum(excess_trucks_per_week[:actual_weeks])
        forecast_excess = np.sum(excess_trucks_per_week[actual_weeks:])
        
        results.append({
            'Scenario': scenario_name,
            'Weeks Over Capacity': int(weeks_over_capacity),
            'Total Excess Truck-Weeks': f"{total_excess_truck_weeks:.1f}",
            'Actual Period Excess': f"{actual_excess:.1f}",
            'Forecast Period Excess': f"{forecast_excess:.1f}",
            'Average Excess (when over)': f"{avg_excess_when_over:.1f}",
            'Peak Excess Trucks': f"{max_excess:.1f}",
            'Week of Peak': int(week_of_max_excess) if week_of_max_excess else '-',
            'Total Annual Rental Cost': f"${total_rental_cost:,.0f}",
            'Cost (numeric)': total_rental_cost
        })
    
    df_costs = pd.DataFrame(results)
    return df_costs


def create_capacity_visualization(all_scenarios, historical_2024, actual_2025, forecast_start_week, 
                                   fleet_capacity=48, colors=None):

    """
    Create a visualization showing total trucks vs fleet capacity with rental cost areas highlighted.
    """
    import plotly.graph_objects as go
    
    hubs = ["Boisbriand", "Châteauguay", "Varennes"]

    if colors is None:
        colors = {
            'Base Forecast': '#95a5a6',
            'Sales +10%': '#3498db',
            'Transactions +10%': '#e74c3c',
            'Sales +10% & Transactions +10%': '#9b59b6'
        }

    # colors = {
    #     'Base Forecast': '#95a5a6',
    #     'Sales +10%': '#3498db',
    #     'Transactions +10%': '#e74c3c',
    #     'Sales +10% & Transactions +10%': '#9b59b6'
    # }
    
    fig = go.Figure()
    
    # Add fleet capacity line
    fig.add_trace(
        go.Scatter(
            x=[0, 52],
            y=[fleet_capacity, fleet_capacity],
            mode='lines',
            name='Fleet Capacity (48 trucks)',
            line=dict(color='green', width=3, dash='dash'),
            hovertemplate='<b>Fleet Capacity</b><br>%{y} trucks<extra></extra>'
        )
    )
    
    # Add shaded area above capacity (rental cost zone)
    fig.add_trace(
        go.Scatter(
            x=[0, 52, 52, 0],
            y=[fleet_capacity, fleet_capacity, 100, 100],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    # Add annotation for rental cost zone
    fig.add_annotation(
        x=26, y=fleet_capacity + 5,
        text="Rental Cost Zone ($4,500/truck/week)",
        showarrow=False,
        font=dict(size=12, color='red'),
        bgcolor='rgba(255, 255, 255, 0.8)',
        borderpad=5
    )
    
    # Add 2024 historical data (total across hubs)
    if historical_2024:
        weeks_2024 = list(range(1, 53))
        total_trucks_2024 = np.zeros(52)
        
        for hub in hubs:
            if hub in historical_2024:
                total_trucks_2024 += historical_2024[hub]
        
        fig.add_trace(
            go.Scatter(
                x=weeks_2024,
                y=total_trucks_2024,
                mode='lines+markers',
                name='2024 Actual',
                line=dict(color='#34495e', width=2, dash='dot'),
                marker=dict(size=4),
                opacity=0.6,
                hovertemplate='<b>2024 Actual</b><br>Week: %{x}<br>Total Trucks: %{y:.1f}<extra></extra>'
            )
        )
    
    # Add 2025 actual data (total across hubs)
    if actual_2025:
        all_weeks = []
        for hub in hubs:
            if hub in actual_2025:
                all_weeks.extend(actual_2025[hub]['weeks'])
        
        if all_weeks:
            unique_weeks = sorted(set(all_weeks))
            total_trucks_2025_actual = []
            
            for week in unique_weeks:
                week_total = 0
                for hub in hubs:
                    if hub in actual_2025 and week in actual_2025[hub]['weeks']:
                        idx = actual_2025[hub]['weeks'].index(week)
                        week_total += actual_2025[hub]['trucks'][idx]
                total_trucks_2025_actual.append(week_total)
            
            fig.add_trace(
                go.Scatter(
                    x=unique_weeks,
                    y=total_trucks_2025_actual,
                    mode='lines+markers',
                    name='2025 Actual',
                    line=dict(color='#2E86AB', width=2.5),
                    marker=dict(size=5),
                    hovertemplate='<b>2025 Actual</b><br>Week: %{x}<br>Total Trucks: %{y:.1f}<extra></extra>'
                )
            )
    
    # Add scenario forecasts
    num_forecast_weeks = len(next(iter(all_scenarios.values()))['Boisbriand'])
    forecast_week_indices = list(range(forecast_start_week, forecast_start_week + num_forecast_weeks))
    
    for scenario_name, scenario_data in all_scenarios.items():
        total_forecast = np.zeros(num_forecast_weeks)
        for hub in hubs:
            total_forecast += scenario_data[hub]
        
        if scenario_name == 'Base Forecast':
            fig.add_trace(
                go.Scatter(
                    x=forecast_week_indices,
                    y=total_forecast,
                    mode='lines+markers',
                    name=scenario_name,
                    line=dict(color=colors[scenario_name], width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Total Trucks: %{{y:.1f}}<extra></extra>'
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=forecast_week_indices,
                    y=total_forecast,
                    mode='lines+markers',
                    name=scenario_name,
                    line=dict(color=colors[scenario_name], width=2.5, dash='dash'),
                    marker=dict(size=5, symbol='square'),
                    hovertemplate=f'<b>{scenario_name}</b><br>Week: %{{x}}<br>Total Trucks: %{{y:.1f}}<extra></extra>'
                )
            )
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=forecast_start_week - 0.5,
        line_dash="dot",
        line_color="red",
        opacity=0.6,
        annotation_text=f"Forecast Start (Week {forecast_start_week})",
        annotation_position="top"
    )
    
    fig.update_layout(
        title={
            'text': 'Fleet Capacity vs Demand - Rental Cost Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title='Week Number',
        yaxis_title='Total Trucks',
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        xaxis=dict(range=[0, 53]),
        yaxis=dict(range=[0, max(fleet_capacity * 1.5, 60)]),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig




def calculate_net_profitability(scenarios, df_rental_costs, base_cost):
    """
    Calculate net profitability considering both revenue changes and rental costs.
    
    Parameters:
    - scenarios: dict with scenario definitions (sales/transactions/outbound changes)
    - df_rental_costs: DataFrame with rental cost analysis
    - base_cost: base forecast rental cost (numeric)
    
    Returns:
    - DataFrame with profitability analysis
    """
    
    # You can adjust these assumptions based on your business
    # Default values - can be made into input parameters
    avg_transaction_value = 150  # Average $ per transaction
    profit_margin = 0.25  # 25% profit margin on sales
    weeks_per_year = 52
    
    profitability_data = []
    
    for _, cost_row in df_rental_costs.iterrows():
        scenario_name = cost_row['Scenario']
        
        if scenario_name == 'Base Forecast':
            # Base forecast has no changes
            profitability_data.append({
                'Scenario': scenario_name,
                'Revenue Impact': '$0',
                'Rental Cost Impact': '$0',
                'Net Impact': '$0',
                'Profitable?': '—'
            })
            continue
        
        # Get scenario changes
        scenario_changes = scenarios[scenario_name]
        sales_change_pct = scenario_changes['sales']
        transactions_change_pct = scenario_changes['transactions']
        
        # Calculate revenue impact
        # Assuming sales change directly impacts revenue
        # You would need to provide actual annual sales figure here
        # For now, we'll estimate based on transactions
        
        # Estimate annual transactions (this is a placeholder - adjust to your data)
        # You might want to calculate this from df_train historical data
        estimated_annual_transactions = 50000  # Placeholder - adjust this!
        
        # Revenue impact from sales change
        revenue_from_sales_change = (estimated_annual_transactions * avg_transaction_value * 
                                     (sales_change_pct / 100) * profit_margin)
        
        # Revenue impact from transaction volume change
        revenue_from_transaction_change = (estimated_annual_transactions * avg_transaction_value * 
                                           (transactions_change_pct / 100) * profit_margin)
        
        # Total revenue impact (profit, not just revenue)
        total_revenue_impact = revenue_from_sales_change + revenue_from_transaction_change
        
        # Rental cost impact
        scenario_cost = cost_row['Cost (numeric)']
        rental_cost_impact = scenario_cost - base_cost
        
        # Net impact = Revenue gain - Additional rental costs
        net_impact = total_revenue_impact - rental_cost_impact
        
        # Determine if profitable
        if net_impact > 0:
            profitable = '✅ Yes'
        elif net_impact < 0:
            profitable = '❌ No'
        else:
            profitable = '➖ Break Even'
        
        profitability_data.append({
            'Scenario': scenario_name,
            'Revenue Impact': f'${total_revenue_impact:+,.0f}',
            'Rental Cost Impact': f'${rental_cost_impact:+,.0f}',
            'Net Impact': f'${net_impact:+,.0f}',
            'Profitable?': profitable,
            'Net Impact (numeric)': net_impact  # For sorting
        })
    
    df_profitability = pd.DataFrame(profitability_data)
    return df_profitability

def calculate_net_profitability_from_ly_sales(scenarios, df_rental_costs, base_cost, df_future):
    """
    Calculate net profitability using LY SALES from df_future (forecast period only).
    Compares revenue change vs rental cost change for the same time period.
    
    Parameters:
    - scenarios: dict with scenario definitions
    - df_rental_costs: DataFrame with rental cost analysis
    - base_cost: base forecast rental cost (numeric)
    - df_future: future dataframe with LY SALES column
    
    Returns:
    - DataFrame with profitability analysis
    """
    
    # Calculate baseline sales from LY SALES for the forecast period
    if 'LY SALES' in df_future.columns:
        # Sum of last year's sales for the forecast period weeks
        baseline_forecast_period_sales = df_future['LY SALES'].sum()
    else:
        baseline_forecast_period_sales = 0
        st.warning("⚠️ LY SALES column not found in future data. Revenue calculation unavailable.")
    
    profitability_data = []
    
    for _, cost_row in df_rental_costs.iterrows():
        scenario_name = cost_row['Scenario']
        
        if scenario_name == 'Base Forecast':
            profitability_data.append({
                'Scenario': scenario_name,
                'Revenue Change': '$0',
                'Rental Cost Change': '$0',
                'Net Impact': '$0',
                'Worth It?': '—'
            })
            continue
        
        # Get scenario changes
        scenario_changes = scenarios[scenario_name]
        sales_change_pct = scenario_changes['sales']
        
        # Calculate revenue change from sales % change
        # New sales = LY SALES × (1 + sales_change_pct)
        # Revenue change = New sales - LY SALES = LY SALES × sales_change_pct
        revenue_change = baseline_forecast_period_sales * (sales_change_pct / 100)
        
        # Rental cost impact (already calculated for the forecast period)
        scenario_cost = cost_row['Cost (numeric)']
        rental_cost_change = scenario_cost - base_cost
        
        # Net impact = Revenue change - Rental cost change
        net_impact = revenue_change - rental_cost_change
        
        # Determine if worth it
        if net_impact > 1000:  # Buffer for rounding
            worth_it = '✅ Yes'
            worth_it_icon = '✅'
        elif net_impact < -1000:
            worth_it = '❌ No'
            worth_it_icon = '❌'
        else:
            worth_it = '➖ Break Even'
            worth_it_icon = '➖'
        
        profitability_data.append({
            'Scenario': scenario_name,
            'Revenue Change': f'${revenue_change:+,.0f}',
            'Rental Cost Change': f'${rental_cost_change:+,.0f}',
            'Net Impact': f'${net_impact:+,.0f}',
            'Worth It?': worth_it,
            'Icon': worth_it_icon,
            'Net Impact (numeric)': net_impact
        })
    
    df_profitability = pd.DataFrame(profitability_data)
    
    # Sort by net impact (best to worst)
    if 'Net Impact (numeric)' in df_profitability.columns:
        df_profitability = df_profitability.sort_values('Net Impact (numeric)', ascending=False)
    
    return df_profitability, baseline_forecast_period_sales


def calculate_net_profitability_with_inputs(scenarios, df_rental_costs, base_cost, 
                                            df_train, profit_margin):
    """
    Calculate net profitability using actual sales data from df_train.
    
    Parameters:
    - scenarios: dict with scenario definitions
    - df_rental_costs: DataFrame with rental cost analysis
    - base_cost: base forecast rental cost (numeric)
    - df_train: training dataframe with actual sales data
    - profit_margin: profit margin as decimal (e.g., 0.25 for 25%)
    
    Returns:
    - DataFrame with profitability analysis
    """
    
    # Calculate baseline annual sales from actual data
    # Use data from the forecast period weeks (weeks that will be forecasted in 2025)
    if 'ACTUAL SALES' in df_train.columns:
        # Get the most recent year of data as baseline
        if 'Year Hiérarchie - Year' in df_train.columns:
            latest_year = df_train['Year Hiérarchie - Year'].max()
            recent_data = df_train[df_train['Year Hiérarchie - Year'] == latest_year]
        else:
            # Use last 52 weeks
            recent_data = df_train.tail(52)
        
        # Calculate weekly average sales
        weekly_avg_sales = recent_data['ACTUAL SALES'].mean()
        
        # Estimate annual sales (52 weeks)
        baseline_annual_sales = weekly_avg_sales * 52
    else:
        # Fallback if ACTUAL SALES column doesn't exist
        baseline_annual_sales = 0
        st.warning("⚠️ ACTUAL SALES column not found in data. Profitability calculation may be inaccurate.")
    
    profitability_data = []
    
    for _, cost_row in df_rental_costs.iterrows():
        scenario_name = cost_row['Scenario']
        
        if scenario_name == 'Base Forecast':
            profitability_data.append({
                'Scenario': scenario_name,
                'Revenue Impact': '$0',
                'Rental Cost Impact': '$0',
                'Net Impact': '$0',
                'Profitable?': '—',
                'ROI': '—'
            })
            continue
        
        # Get scenario changes
        scenario_changes = scenarios[scenario_name]
        sales_change_pct = scenario_changes['sales']
        transactions_change_pct = scenario_changes['transactions']
        
        # Calculate revenue impact using actual sales data
        # Sales % change directly affects revenue
        # Transactions % change affects volume (which also affects revenue)
        
        # Method: Combined multiplicative effect
        # If sales +10% (price/mix) and transactions +10% (volume)
        # Total revenue impact = baseline * (1 + sales%) * (1 + transactions%)
        
        sales_multiplier = 1 + (sales_change_pct / 100)
        transaction_multiplier = 1 + (transactions_change_pct / 100)
        
        # Calculate base annual profit
        base_annual_profit = baseline_annual_sales * profit_margin
        
        # Calculate new annual sales with scenario changes
        new_annual_sales = baseline_annual_sales * sales_multiplier * transaction_multiplier
        
        # Calculate new annual profit
        new_annual_profit = new_annual_sales * profit_margin
        
        # Revenue impact (additional profit)
        total_revenue_impact = new_annual_profit - base_annual_profit
        
        # Rental cost impact
        scenario_cost = cost_row['Cost (numeric)']
        rental_cost_impact = scenario_cost - base_cost
        
        # Net impact = Additional profit - Additional rental costs
        net_impact = total_revenue_impact - rental_cost_impact
        
        # Calculate ROI
        if rental_cost_impact != 0:
            roi = (net_impact / abs(rental_cost_impact)) * 100
            roi_str = f'{roi:+.1f}%'
        else:
            roi_str = '∞' if net_impact > 0 else '—'
        
        # Determine if profitable
        if net_impact > 1000:  # Buffer for rounding
            profitable = '✅ Profitable'
        elif net_impact < -1000:
            profitable = '❌ Not Profitable'
        else:
            profitable = '➖ Break Even'
        
        profitability_data.append({
            'Scenario': scenario_name,
            'Revenue Impact': f'${total_revenue_impact:+,.0f}',
            'Rental Cost Impact': f'${rental_cost_impact:+,.0f}',
            'Net Impact': f'${net_impact:+,.0f}',
            'ROI': roi_str,
            'Profitable?': profitable,
            'Net Impact (numeric)': net_impact
        })
    
    df_profitability = pd.DataFrame(profitability_data)
    
    # Sort by net impact (best to worst)
    if 'Net Impact (numeric)' in df_profitability.columns:
        df_profitability = df_profitability.sort_values('Net Impact (numeric)', ascending=False)
    
    return df_profitability


# Main app logic
try:
    # Load data
    with st.spinner("Loading data..."):
        df_train, df_future = load_data()
        models = load_models()
    
    st.success("✅ Data and models loaded successfully!")
    
    # Show data info
    with st.expander("📊 Data Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Data Rows", len(df_train))
        with col2:
            st.metric("Future Forecast Weeks", len(df_future))
    
    # Add tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 What-If Scenarios", "📅 Year-over-Year Comparison", "🔮 2026 Predictions"])

    
    with tab1:
            # NEW: Add scenario builder UI BEFORE the button
        st.markdown("### 🎯 Step 1: Define Your Scenarios")
        scenarios = create_dynamic_scenarios_ui()
        
        st.markdown("---")
        st.markdown("### 🚀 Step 2: Generate Analysis")
        # Button to generate forecast
        if st.button("🚀 Generate What-If Scenarios", type="primary", key="whatif"):
            try:
                st.write("**DEBUG 1: Button clicked**")
                st.write("**DEBUG 2: Scenarios from UI:**")
                st.write(scenarios)
                st.write(f"**DEBUG 3: Number of scenarios: {len(scenarios)}**")
                st.write(f"**DEBUG 4: Scenario keys: {list(scenarios.keys())}**")
                
                with st.spinner("Calculating elasticity and generating scenarios..."):
                    st.write("**DEBUG 5: Entering spinner**")
                    
                    # Apply seasonal adjustments first
                    st.write("**DEBUG 6: Applying seasonal adjustments**")
                    df_future_adjusted, adjustment_factors = apply_seasonal_adjustment_to_forecast(
                        df_train, df_future, models
                    )
                    st.write("**DEBUG 7: Seasonal adjustments done**")
                    
                    # Calculate elasticity coefficients
                    st.write("**DEBUG 8: Calculating elasticity**")
                    elasticities = calculate_elasticity_coefficients(df_train)
                    st.write("**DEBUG 9: Elasticity done**")

                    # Calculate confidence intervals
                    st.write("**DEBUG 10: Calculating confidence intervals**")
                    intervals = calculate_confidence_intervals(df_train, models, confidence_level=0.95)
                    st.write("**DEBUG 11: Confidence intervals done**")

                    # Create scenarios using dynamic user inputs
                    st.write("**DEBUG 12: Creating dynamic scenarios**")
                    st.write(f"**DEBUG 13: Passing scenarios: {list(scenarios.keys())}**")
                    all_scenarios, historical_2024, actual_2025, forecast_start_week = create_dynamic_whatif_scenarios(
                        df_train, df_future_adjusted, models, elasticities, scenarios
                    )
                    st.write("**DEBUG 14: Dynamic scenarios created**")
                    st.write(f"**DEBUG 15: all_scenarios keys: {list(all_scenarios.keys())}**")
                
                    # Get dynamic colors for scenarios
                    st.write("**DEBUG 16: Getting colors**")
                    colors = get_scenario_colors(scenarios)
                    st.write(f"**DEBUG 17: Colors: {colors}**")
                    
                    # Display summary statistics
                    st.write("**DEBUG 18: Creating summary statistics**")
                    st.subheader("📈 Scenario Summary")
                    
                    summary_data = []
                    for scenario_name, scenario_data in all_scenarios.items():
                        for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
                            avg_trucks = scenario_data[hub].mean()
                            summary_data.append({
                                'Scenario': scenario_name,
                                'Hub': hub,
                                'Avg Trucks': f"{avg_trucks:.2f}"
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_pivot = summary_df.pivot(index='Scenario', columns='Hub', values='Avg Trucks')
                    st.dataframe(summary_pivot, use_container_width=True)
                    
                    st.write("**DEBUG 19: About to plot scenarios**")
                    # Plot scenarios - MODIFIED: Pass intervals
                    st.subheader("📊 Visual Comparison")
                    st.info(f"📍 Forecast starts at week {forecast_start_week} | 🔧 Seasonal adjustments applied to weeks 47-48")
                    fig = plot_scenarios(all_scenarios, historical_2024, actual_2025, forecast_start_week, intervals, colors)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("**DEBUG 20: Scenarios plotted successfully**")

                    # NEW: Add combined plot showing total across all hubs
                    st.subheader("🚛 Total Trucks Across All Hubs")
                    st.markdown("**This chart shows the combined total of trucks needed across Boisbriand, Châteauguay, and Varennes**")
                    fig_combined = plot_scenarios_combined(all_scenarios, intervals, historical_2024, actual_2025, forecast_start_week, colors)
                    st.plotly_chart(fig_combined, use_container_width=True)


                    # After the combined plot and summary statistics, add this:

                    # ============ NEW: RENTAL COST ANALYSIS ============
                    st.markdown("---")
                    st.header("💰 Truck Rental Cost Analysis")
                    st.markdown("""
                    **Fleet Capacity:** 48 trucks total across all hubs  
                    **Rental Cost:** $4,500 per truck per week (when exceeding capacity)  
                    **Analysis Period:** Full year 2025 (actuals + forecast)
                    """)

                    # Calculate annual rental costs
                    df_rental_costs = calculate_annual_rental_costs(
                        all_scenarios, historical_2024, actual_2025, forecast_start_week,
                        fleet_capacity=48, cost_per_week=4500
                    )

                    # Display cost comparison table
                    st.subheader("📊 Rental Cost Comparison by Scenario")
                    display_df = df_rental_costs.drop(columns=['Cost (numeric)'])
                    st.dataframe(display_df, use_container_width=True)

                    # Highlight base forecast costs
                    base_forecast_row = df_rental_costs[df_rental_costs['Scenario'] == 'Base Forecast'].iloc[0]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Weeks Over Capacity",
                            base_forecast_row['Weeks Over Capacity'],
                            help="Number of weeks where total demand exceeds 48 trucks"
                        )
                    with col2:
                        st.metric(
                            "Total Excess Truck-Weeks",
                            base_forecast_row['Total Excess Truck-Weeks'],
                            help="Sum of all excess trucks across all weeks"
                        )
                    with col3:
                        st.metric(
                            "Peak Excess",
                            f"{base_forecast_row['Peak Excess Trucks']} trucks",
                            help=f"Maximum trucks over capacity (Week {base_forecast_row['Week of Peak']})"
                        )
                    with col4:
                        st.metric(
                            "Total Annual Cost",
                            base_forecast_row['Total Annual Rental Cost'],
                            help="Total rental cost for 2025"
                        )

                    # Create capacity visualization
                    st.subheader("📈 Fleet Capacity vs Demand")
                    fig_capacity = create_capacity_visualization(
                        all_scenarios, historical_2024, actual_2025, forecast_start_week,
                        fleet_capacity=48, colors=colors
                    )
                    st.plotly_chart(fig_capacity, use_container_width=True)

                    # Cost breakdown by period
                    st.subheader("💵 Cost Breakdown: Actual vs Forecast")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Actual Period (Weeks 1-{}):**".format(forecast_start_week - 1))
                        actual_cost_data = []
                        for _, row in df_rental_costs.iterrows():
                            actual_excess = float(row['Actual Period Excess'])
                            actual_cost = actual_excess * 4500
                            actual_cost_data.append({
                                'Scenario': row['Scenario'],
                                'Excess Truck-Weeks': row['Actual Period Excess'],
                                'Cost': f"${actual_cost:,.0f}"
                            })
                        st.dataframe(pd.DataFrame(actual_cost_data), use_container_width=True)

                    with col2:
                        st.markdown("**Forecast Period (Weeks {}-52):**".format(forecast_start_week))
                        forecast_cost_data = []
                        for _, row in df_rental_costs.iterrows():
                            forecast_excess = float(row['Forecast Period Excess'])
                            forecast_cost = forecast_excess * 4500
                            forecast_cost_data.append({
                                'Scenario': row['Scenario'],
                                'Excess Truck-Weeks': row['Forecast Period Excess'],
                                'Cost': f"${forecast_cost:,.0f}"
                            })
                        st.dataframe(pd.DataFrame(forecast_cost_data), use_container_width=True)

                    # Cost savings comparison
                    st.subheader("💡 Potential Cost Savings")
                    base_cost = df_rental_costs[df_rental_costs['Scenario'] == 'Base Forecast']['Cost (numeric)'].values[0]

                    savings_data = []
                    for _, row in df_rental_costs.iterrows():
                        if row['Scenario'] != 'Base Forecast':
                            scenario_cost = row['Cost (numeric)']
                            savings = base_cost - scenario_cost
                            savings_pct = (savings / base_cost * 100) if base_cost > 0 else 0
                            
                            savings_data.append({
                                'Scenario': row['Scenario'],
                                'Total Cost': row['Total Annual Rental Cost'],
                                'vs Base Forecast': f"${savings:+,.0f} ({savings_pct:+.1f}%)"
                            })

                    if savings_data:
                        st.dataframe(pd.DataFrame(savings_data), use_container_width=True)
                        
                        if base_cost > 0:
                            st.warning(f"⚠️ **Base Forecast Annual Rental Cost: {base_forecast_row['Total Annual Rental Cost']}**")
                            st.info("💡 Consider optimizing operations or adjusting business strategies to reduce weeks over capacity and minimize rental costs.")

                    # ============ END RENTAL COST ANALYSIS ============
                    
                    
                                    # ============ NET PROFITABILITY ANALYSIS ============
                    st.markdown("---")
                    st.header("💰 Revenue vs Cost Analysis")
                    st.markdown("""
                    This analysis answers: **"Is the additional revenue from sales growth worth the extra truck rental costs?"**

                    **Methodology:**
                    - **Revenue Change**: Calculated from LY SALES (last year's sales for the forecast period) × Sales % change
                    - **Rental Cost Change**: Additional truck rental costs for the forecast period
                    - **Net Impact**: Revenue Change - Rental Cost Change
                    """)

                    # Show baseline sales information
                    if 'LY SALES' in df_future.columns:
                        baseline_ly_sales = df_future['LY SALES'].sum()
                        num_forecast_weeks = len(df_future)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Forecast Period", f"{num_forecast_weeks} weeks")
                        with col2:
                            st.metric("LY Sales (Forecast Period)", f"${baseline_ly_sales:,.0f}")
                        with col3:
                            st.metric("Weekly Avg (LY)", f"${baseline_ly_sales/num_forecast_weeks:,.0f}")
                    else:
                        st.warning("⚠️ LY SALES column not found in future data. Revenue analysis unavailable.")
                        baseline_ly_sales = 0

                    # Calculate profitability using LY SALES
                    df_profitability, baseline_sales = calculate_net_profitability_from_ly_sales(
                        scenarios, df_rental_costs, base_cost, df_future
                    )

                    # Display profitability analysis
                    st.subheader("📊 Scenario Comparison")
                    st.markdown("""
                    **How to read this table:**
                    - **Revenue Change**: How much more/less revenue from sales % change (based on LY SALES)
                    - **Rental Cost Change**: How much more/less spent on truck rentals
                    - **Net Impact**: Revenue Change - Rental Cost Change
                    - **Worth It?**: ✅ = Revenue gain exceeds costs | ❌ = Costs exceed revenue gain
                    """)

                    # Display the dataframe
                    display_profitability = df_profitability.drop(columns=['Net Impact (numeric)', 'Icon'], errors='ignore')
                    st.dataframe(display_profitability, use_container_width=True)

                    # Show calculation example for transparency
                    with st.expander("📖 Example Calculation"):
                        st.markdown(f"""
                        **Example for a scenario with +15% Sales:**
                        
                        1. **LY Sales (Forecast Period)**: ${baseline_sales:,.0f}
                        2. **New Sales with +15%**: ${baseline_sales:,.0f} × 1.15 = ${baseline_sales * 1.15:,.0f}
                        3. **Revenue Change**: ${baseline_sales * 1.15:,.0f} - ${baseline_sales:,.0f} = **${baseline_sales * 0.15:,.0f}**
                        4. **Rental Cost Change**: (Calculated from truck needs above/below 48 capacity)
                        5. **Net Impact**: Revenue Change - Rental Cost Change
                        
                        **Decision:** If Net Impact is positive (✅), the sales growth more than pays for the extra trucks!
                        """)

                    # Show best and worst scenarios with icons
                    st.subheader("🏆 Quick Summary")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**💚 Best Financial Outcome:**")
                        best_scenario = df_profitability.iloc[0] if len(df_profitability) > 1 else None
                        if best_scenario is not None and best_scenario['Scenario'] != 'Base Forecast':
                            icon = best_scenario.get('Icon', '✅')
                            st.success(f"""
                            {icon} **{best_scenario['Scenario']}**
                            - Revenue Change: {best_scenario['Revenue Change']}
                            - Rental Cost: {best_scenario['Rental Cost Change']}
                            - **Net: {best_scenario['Net Impact']}**
                            """)
                        else:
                            st.info("Base Forecast is the reference scenario")

                    with col2:
                        st.markdown("**⚠️ Worst Financial Outcome:**")
                        worst_scenario = df_profitability.iloc[-1] if len(df_profitability) > 1 else None
                        if worst_scenario is not None and worst_scenario['Scenario'] != 'Base Forecast':
                            icon = worst_scenario.get('Icon', '❌')
                            st.error(f"""
                            {icon} **{worst_scenario['Scenario']}**
                            - Revenue Change: {worst_scenario['Revenue Change']}
                            - Rental Cost: {worst_scenario['Rental Cost Change']}
                            - **Net: {worst_scenario['Net Impact']}**
                            """)
                        else:
                            st.info("All scenarios have similar outcomes")

                    # Show business insights
                    st.markdown("---")
                    st.subheader("💡 Key Insights")

                    worth_it_scenarios = df_profitability[
                        (df_profitability['Worth It?'] == '✅ Yes') & 
                        (df_profitability['Scenario'] != 'Base Forecast')
                    ]

                    not_worth_it_scenarios = df_profitability[
                        (df_profitability['Worth It?'] == '❌ No') & 
                        (df_profitability['Scenario'] != 'Base Forecast')
                    ]

                    col1, col2 = st.columns(2)

                    with col1:
                        if len(worth_it_scenarios) > 0:
                            st.success(f"✅ **{len(worth_it_scenarios)} scenario(s) are worth it**")
                            st.markdown("Revenue gains exceed rental costs")
                        else:
                            st.info("No scenarios generate net positive revenue")

                    with col2:
                        if len(not_worth_it_scenarios) > 0:
                            st.error(f"❌ **{len(not_worth_it_scenarios)} scenario(s) are not worth it**")
                            st.markdown("Rental costs exceed revenue gains")
                        else:
                            st.success("All scenarios are financially viable")

                    # ============ END NET PROFITABILITY ANALYSIS ============
                    # NEW: Show confidence interval statistics
                    with st.expander("📊 Confidence Interval Statistics (95%)"):
                        for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
                            st.markdown(f"**{hub}:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Std Error", f"{intervals[hub]['std_error']:.2f} trucks")
                            with col2:
                                st.metric("95% CI Margin", f"±{intervals[hub]['margin']:.2f} trucks")
                            with col3:
                                st.metric("Mean Abs Error", f"{intervals[hub]['mean_absolute_error']:.2f} trucks")
            
            
                    # Show elasticity coefficients
                    with st.expander("🔍 Elasticity Coefficients (How metrics affect trucks)"):
                        for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
                            st.markdown(f"**{hub}:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sales Elasticity", f"{elasticities[hub]['sales']:.4f}")
                            with col2:
                                st.metric("Transactions Elasticity", f"{elasticities[hub]['transactions']:.4f}")
                            with col3:
                                st.metric("Outbound Elasticity", f"{elasticities[hub]['outbound']:.4f}")
                    
                    # Show seasonal adjustments applied
                    with st.expander("🔧 Seasonal Adjustments Applied (Weeks 47-48)"):
                        for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
                            factor = adjustment_factors[hub]
                            pct_change = (factor - 1) * 100
                            st.markdown(f"**{hub}:** {factor:.4f} ({pct_change:+.2f}%)")
                    
                    st.success("✅ Analysis complete!")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.write("**Full error details:**")
                import traceback
                st.code(traceback.format_exc())
                st.write("**DEBUG: Last successful step before error**")
    
                
    with tab2:
        st.markdown("### Compare 2024 actual performance vs 2025 (actual + forecast)")
        
        if st.button("📊 Generate Year-over-Year Comparison", type="primary", key="yoy"):
            with st.spinner("Applying seasonal adjustments and creating comparison..."):
                # Apply seasonal adjustments
                df_future_adjusted, adjustment_factors = apply_seasonal_adjustment_to_forecast(
                    df_train, df_future, models
                )
                
                # Create YoY comparison
                fig_yoy, df_comparison = create_yoy_comparison(df_train, df_future_adjusted, models)
                
                # Display the plot
                st.plotly_chart(fig_yoy, use_container_width=True)
                
                # Show the comparison dataframe
                st.subheader("📋 Detailed Comparison Data")
                st.dataframe(df_comparison, use_container_width=True)
                
                # Download button for the comparison data
                csv = df_comparison.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Data as CSV",
                    data=csv,
                    file_name="yoy_comparison_2024_vs_2025.csv",
                    mime="text/csv"
                )
                
                # Show seasonal adjustments applied
                with st.expander("🔧 Seasonal Adjustments Applied"):
                    st.markdown("**Adjustment factors used for weeks 47-48:**")
                    for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
                        factor = adjustment_factors[hub]
                        pct_change = (factor - 1) * 100
                        st.markdown(f"**{hub}:** {factor:.4f} ({pct_change:+.2f}%)")
                        
    with tab3:
        st.markdown("### 🔮 2026 Truck Demand Predictions")
        st.markdown("View predictions for all 52 weeks of 2026 across different scenarios")
        
        if st.button("🚀 Generate 2026 Predictions", type="primary", key="predictions_2026"):
            with st.spinner("Generating 2026 predictions..."):
                try:
                    # Calculate elasticity coefficients
                    elasticities = calculate_elasticity_coefficients(df_train)
                    
                    # Calculate confidence intervals
                    intervals = calculate_confidence_intervals(df_train, models, confidence_level=0.95)
                    
                    # Use default scenarios instead of UI (to avoid duplicate widgets)
                    scenarios = {
                        'Base Forecast': {'sales': 0, 'transactions': 0, 'outbound': 0},
                        'Sales +10%': {'sales': 10, 'transactions': 0, 'outbound': 0},
                        'Transactions +10%': {'sales': 0, 'transactions': 10, 'outbound': 0},
                        'Sales +10% & Transactions +10%': {'sales': 10, 'transactions': 10, 'outbound': 0},
                    }
                    
                    # Generate predictions for extended future (including 2026)
                    all_scenarios, historical_2024, actual_2025, forecast_start_week = create_dynamic_whatif_scenarios(
                        df_train, df_future, models, elasticities, scenarios
                    )
                    
                    # Get colors
                    colors = get_scenario_colors(scenarios)
                    
                    # Filter to 2026 only
                    st.subheader("📊 2026 Predictions by Hub")
                    
                    # Display 2026 predictions plot
                    fig_2026 = plot_2026_predictions(all_scenarios, forecast_start_week, colors)
                    st.plotly_chart(fig_2026, use_container_width=True)
                    
                    # Create downloadable table of 2026 predictions
                    st.subheader("📥 Download 2026 Predictions")
                    
                    # Build dataframe
                    prediction_data = {'Week': list(range(1, 53))}
                    
                    for scenario_name, scenario_data in all_scenarios.items():
                        for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
                            forecast = scenario_data[hub]
                            # Get last 52 weeks (2026)
                            if len(forecast) >= 52:
                                forecast_2026 = forecast[-52:]
                            else:
                                forecast_2026 = list(forecast) + [0] * (52 - len(forecast))
                            
                            prediction_data[f'{hub}_{scenario_name}'] = forecast_2026
                    
                    df_2026_predictions = pd.DataFrame(prediction_data)
                    
                    # Display table
                    st.dataframe(df_2026_predictions, use_container_width=True)
                    
                    # Download button
                    csv = df_2026_predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download 2026 Predictions as CSV",
                        data=csv,
                        file_name="2026_truck_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Also create a text file
                    txt_content = "="*70 + "\n"
                    txt_content += "TRUCK PREDICTIONS FOR 2026\n"
                    txt_content += "="*70 + "\n\n"
                    
                    for hub in ["Boisbriand", "Châteauguay", "Varennes"]:
                        txt_content += f"\n{hub}:\n"
                        txt_content += "-"*50 + "\n"
                        
                        for scenario_name in scenarios.keys():
                            txt_content += f"\n  {scenario_name}:\n"
                            forecast = scenario_data[hub][-52:] if len(scenario_data[hub]) >= 52 else scenario_data[hub]
                            
                            for week, trucks in enumerate(forecast, 1):
                                txt_content += f"    Week {week:2d}: {trucks:.1f} trucks\n"
                    
                    txt_content += "\n" + "="*70 + "\n"
                    
                    st.download_button(
                        label="📥 Download 2026 Predictions as TXT",
                        data=txt_content,
                        file_name="2026_truck_predictions.txt",
                        mime="text/plain"
                    )
                    
                    st.success("✅ 2026 predictions generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

except FileNotFoundError as e:
    st.error(f"❌ Error: Could not find required files. Make sure you have:")
    st.code("- df_train_streamlit.xlsx\n- X_future_streamlit.xlsx\n- models.pkl")
    st.info("💡 Run your Jupyter notebook first to generate these files")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("Please check your data files and model format")