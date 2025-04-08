# pages/6_measures_of_association.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add path to import the code lab module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interactive_code'))
import measure_of_association_code

########################
# Utility / Calculation #
########################
def calculate_measures(exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases):
    """Calculate various measures of association."""
    # Risk in exposed
    risk_exposed = exposed_cases / (exposed_cases + exposed_noncases) if (exposed_cases + exposed_noncases) > 0 else 0
    # Risk in unexposed
    risk_unexposed = unexposed_cases / (unexposed_cases + unexposed_noncases) if (unexposed_cases + unexposed_noncases) > 0 else 0

    # Risk Ratio
    risk_ratio = risk_exposed / risk_unexposed if risk_unexposed != 0 else float('inf')

    # Odds in exposed and unexposed
    odds_exposed = exposed_cases / exposed_noncases if exposed_noncases != 0 else float('inf')
    odds_unexposed = unexposed_cases / unexposed_noncases if unexposed_noncases != 0 else float('inf')

    # Odds Ratio
    odds_ratio = odds_exposed / odds_unexposed if odds_unexposed != 0 else float('inf')

    # Risk Difference
    risk_difference = risk_exposed - risk_unexposed

    return {
        'Risk Ratio': risk_ratio,
        'Odds Ratio': odds_ratio,
        'Risk Difference': risk_difference,
        'Risk in Exposed': risk_exposed,
        'Risk in Unexposed': risk_unexposed,
        'Odds in Exposed': odds_exposed,
        'Odds in Unexposed': odds_unexposed
    }

def display_2x2_table():
    """Display an editable 2x2 contingency table and return the values."""
    # Using a more stable approach for the 2x2 table
    st.markdown("Click on any cell to edit the values:")
    
    # Use form to make the table more stable
    with st.form(key="contingency_table_form"):
        # First row - Exposed
        col1, col2 = st.columns(2)
        with col1:
            exposed_cases = st.number_input(
                "Exposed Cases",
                min_value=0,
                value=40,
                step=1,
                format="%d",
                key="exposed_cases"
            )
        with col2:
            exposed_noncases = st.number_input(
                "Exposed Non-cases",
                min_value=0,
                value=60,
                step=1,
                format="%d",
                key="exposed_noncases"
            )
        
        # Second row - Unexposed
        col3, col4 = st.columns(2)
        with col3:
            unexposed_cases = st.number_input(
                "Unexposed Cases",
                min_value=0,
                value=20,
                step=1,
                format="%d",
                key="unexposed_cases"
            )
        with col4:
            unexposed_noncases = st.number_input(
                "Unexposed Non-cases",
                min_value=0,
                value=80,
                step=1,
                format="%d",
                key="unexposed_noncases"
            )
        
        # Submit button
        submit_button = st.form_submit_button(label="Update Results")
    
    # For display, create a DataFrame
    contingency_df = pd.DataFrame({
        'Cases': [exposed_cases, unexposed_cases],
        'Non-cases': [exposed_noncases, unexposed_noncases]
    }, index=['Exposed', 'Unexposed'])
    
    # Display the table (not editable, just for visualization)
    st.table(contingency_df)
    
    return exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases

def risk_based_measures_tab(exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases):
    """Display content for the Risk-Based Measures tab."""
    st.header("Risk-Based Measures")
    
    # Calculate measures
    measures = calculate_measures(
        exposed_cases, exposed_noncases,
        unexposed_cases, unexposed_noncases
    )
    
    # Display risk-based results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk in Exposed", f"{measures['Risk in Exposed']:.3f}")
        st.metric("Risk Ratio (RR)", f"{measures['Risk Ratio']:.3f}")
    
    with col2:
        st.metric("Risk in Unexposed", f"{measures['Risk in Unexposed']:.3f}")
        st.metric("Risk Difference (RD)", f"{measures['Risk Difference']:.3f}")
    
    # Visualization for risk-based measures
    data = pd.DataFrame({
        'Group': ['Exposed', 'Unexposed'],
        'Risk': [measures['Risk in Exposed'], measures['Risk in Unexposed']]
    })
    
    fig = px.bar(
        data,
        x='Group',
        y='Risk',
        color='Group',
        title='Risk Comparison Between Exposed and Unexposed Groups',
        labels={'Risk': 'Risk (Probability of Disease)'},
        color_discrete_map={'Exposed': '#636EFA', 'Unexposed': '#EF553B'}
    )
    
    # Add a horizontal line for the risk ratio
    fig.add_shape(
        type="line",
        x0=-0.5, x1=0.5,
        y0=measures['Risk in Exposed'], y1=measures['Risk in Exposed'],
        line=dict(color="green", width=2, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=0.5, x1=1.5,
        y0=measures['Risk in Unexposed'], y1=measures['Risk in Unexposed'],
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Add annotation for risk ratio
    fig.add_annotation(
        x=0.5,
        y=(measures['Risk in Exposed'] + measures['Risk in Unexposed']) / 2,
        text=f"RR = {measures['Risk Ratio']:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Add annotation for risk difference
    fig.add_annotation(
        x=0.5,
        y=measures['Risk in Unexposed'] + measures['Risk Difference'] / 2,
        text=f"RD = {measures['Risk Difference']:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=40
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Educational content for risk-based measures
    st.subheader("Understanding Risk-Based Measures")
    
    st.markdown("""
    ### Risk (Incidence Proportion)
    **Risk** is the probability of developing the disease during a specified time period.
    - Formula: Number of new cases / Total population at risk
    - Range: 0 to 1 (or 0% to 100%)
    
    ### Risk Ratio (Relative Risk)
    The **Risk Ratio (RR)** is the ratio of the risk in the exposed group to the risk in the unexposed group.
    - Formula: Risk in exposed / Risk in unexposed
    - Interpretation:
        - **RR = 1**: No association between exposure and disease
        - **RR > 1**: Exposure associated with increased risk
        - **RR < 1**: Exposure associated with decreased risk (protective effect)
    
    ### Risk Difference (Attributable Risk)
    The **Risk Difference (RD)** is the absolute difference in risk between exposed and unexposed groups.
    - Formula: Risk in exposed - Risk in unexposed
    - Interpretation:
        - **RD > 0**: Exposure increases risk by RD amount
        - **RD = 0**: No difference in risk
        - **RD < 0**: Exposure decreases risk by RD amount
    - Used to calculate Number Needed to Treat (NNT) or Number Needed to Harm (NNH): 1/|RD|
    """)
    
    st.markdown("""
    ### When to Use Risk-Based Measures
    - **Cohort studies**: Where you follow participants over time to observe outcomes
    - **Randomized controlled trials**: Comparing treatment to control groups
    - **Public health planning**: When absolute impact matters
    - **Communication**: When explaining risks to the general public
    
    ### Limitations
    - Cannot be directly calculated in case-control studies
    - May not account for confounding without adjustment
    - Time-dependent risks need careful interpretation
    """)
    
    # Links to additional resources
    st.subheader("Additional Resources")
    st.markdown("""
    - [Link 1]
    - [Link 2] 
    - [Link 3]
    """)

def odds_based_measures_tab(exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases):
    """Display content for the Odds-Based Measures tab."""
    st.header("Odds-Based Measures")
    
    # Calculate measures
    measures = calculate_measures(
        exposed_cases, exposed_noncases,
        unexposed_cases, unexposed_noncases
    )
    
    # Display odds-based results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Odds in Exposed", f"{measures['Odds in Exposed']:.3f}")
        st.metric("Odds Ratio (OR)", f"{measures['Odds Ratio']:.3f}")
    
    with col2:
        st.metric("Odds in Unexposed", f"{measures['Odds in Unexposed']:.3f}")
        st.metric("Risk Ratio (for comparison)", f"{measures['Risk Ratio']:.3f}")
    
    # Visualization for odds-based measures
    data = pd.DataFrame({
        'Group': ['Exposed', 'Unexposed'],
        'Odds': [measures['Odds in Exposed'], measures['Odds in Unexposed']]
    })
    
    fig = px.bar(
        data,
        x='Group',
        y='Odds',
        color='Group',
        title='Odds Comparison Between Exposed and Unexposed Groups',
        labels={'Odds': 'Odds of Disease (Cases/Non-cases)'},
        color_discrete_map={'Exposed': '#636EFA', 'Unexposed': '#EF553B'}
    )
    
    # Add a horizontal line for the odds values
    fig.add_shape(
        type="line",
        x0=-0.5, x1=0.5,
        y0=measures['Odds in Exposed'], y1=measures['Odds in Exposed'],
        line=dict(color="green", width=2, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=0.5, x1=1.5,
        y0=measures['Odds in Unexposed'], y1=measures['Odds in Unexposed'],
        line=dict(color="green", width=2, dash="dash"),
    )
    
    # Add annotation for odds ratio
    fig.add_annotation(
        x=0.5,
        y=(measures['Odds in Exposed'] + measures['Odds in Unexposed']) / 2,
        text=f"OR = {measures['Odds Ratio']:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison between OR and RR
    if not np.isinf(measures['Odds Ratio']) and not np.isinf(measures['Risk Ratio']):
        st.subheader("Comparison: Odds Ratio vs. Risk Ratio")
        
        # Create a comparison DataFrame
        comparison_df = pd.DataFrame({
            'Measure': ['Odds Ratio', 'Risk Ratio'],
            'Value': [measures['Odds Ratio'], measures['Risk Ratio']]
        })
        
        fig2 = px.bar(
            comparison_df,
            x='Measure',
            y='Value',
            color='Measure',
            title='Odds Ratio vs. Risk Ratio for This Dataset',
            labels={'Value': 'Magnitude of Association'},
            color_discrete_map={'Odds Ratio': '#AB63FA', 'Risk Ratio': '#FFA15A'}
        )
        
        # Calculate percent difference
        percent_diff = (measures['Odds Ratio'] - measures['Risk Ratio']) / measures['Risk Ratio'] * 100
        
        # Add annotation about relationship
        fig2.add_annotation(
            x=0.5,
            y=max(measures['Odds Ratio'], measures['Risk Ratio']) * 1.1,
            text=f"OR is {abs(percent_diff):.1f}% {'higher' if percent_diff > 0 else 'lower'} than RR",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Educational content for odds-based measures
    st.subheader("Understanding Odds-Based Measures")
    
    st.markdown("""
    ### Odds
    **Odds** represent the ratio of the probability of occurrence to the probability of non-occurrence.
    - Formula: Number of cases / Number of non-cases
    - Range: 0 to infinity
    - Conversion to probability: Odds / (1 + Odds)
    
    ### Odds Ratio (OR)
    The **Odds Ratio (OR)** is the ratio of the odds of disease in the exposed group to the odds of disease in the unexposed group.
    - Formula: Odds in exposed / Odds in unexposed
    - Interpretation:
        - **OR = 1**: No association between exposure and disease
        - **OR > 1**: Exposure associated with increased odds of disease
        - **OR < 1**: Exposure associated with decreased odds of disease
    
    ### Relationship Between OR and RR
    - When the disease is rare (low incidence), OR approximates RR
    - As disease prevalence increases, OR overestimates RR for risk factors (OR > RR)
    - For protective factors, OR underestimates RR (OR closer to 1 than RR)
    """)
    
    st.markdown("""
    ### When to Use Odds-Based Measures
    - **Case-control studies**: Where you cannot directly calculate incidence/risk
    - **Logistic regression**: OR is the natural output of this common analysis method
    - **Meta-analyses**: Often use OR because it can be estimated from various study designs
    - **Rare diseases**: When OR closely approximates RR
    
    ### Limitations
    - Less intuitive to interpret than risk-based measures
    - Can overestimate the effect size when the outcome is common
    - May be misinterpreted if presented as equivalent to RR
    """)
    
    # Links to additional resources
    st.subheader("Additional Resources")
    st.markdown("""
    - [BMJ: Odds ratios](https://www.bmj.com/content/320/7247/1468)
    - [StatsDirect: Odds ratio vs Relative risk](https://www.statsdirect.com/help/basics/oddsratio.htm)
    - [MedCalc: Odds ratio calculator](https://www.medcalc.org/calc/odds_ratio.php)
    """)

###################################
# Streamlit Layout Configuration #
###################################
st.set_page_config(layout="wide")

st.title("Measures of Association in Epidemiology")

# Main tabs for the interactive visualization vs code lab
main_tabs = st.tabs(["📊 Interactive Visualization", "💻 Code Laboratory"])

# Interactive Visualization tab
with main_tabs[0]:
    # Create 2x2 table once to avoid shakiness
    st.header("2x2 Contingency Table")
    exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases = display_2x2_table()
    
    # Create tabs for risk-based vs odds-based measures
    measure_tabs = st.tabs(["⚠️Risk-Based Measures", "⚖️Odds-Based Measures"])
    
    # Populate Risk-Based Measures tab
    with measure_tabs[0]:
        risk_based_measures_tab(exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases)
    
    # Populate Odds-Based Measures tab
    with measure_tabs[1]:
        odds_based_measures_tab(exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases)

# Code Lab tab
with main_tabs[1]:
    measure_of_association_code.app()