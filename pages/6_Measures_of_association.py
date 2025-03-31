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
    risk_exposed = exposed_cases / (exposed_cases + exposed_noncases)
    # Risk in unexposed
    risk_unexposed = unexposed_cases / (unexposed_cases + unexposed_noncases)

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
        'Risk in Unexposed': risk_unexposed
    }

###################################
# Streamlit Layout Configuration #
###################################
st.set_page_config(layout="wide")

st.title("Measures of Association in Epidemiology")
viz_tab, code_tab = st.tabs(["📊Interactive Visualization", "💻 Code Laboratory"])

# Wrap your existing visualization content in:
with viz_tab:
    ###############################
    # 2x2 Table Input & Controls  #
    ###############################
    st.header("2x2 Contingency Table")
    st.markdown("Click on any cell to edit the values:")

    # Create a default 2x2 table as a DataFrame
    default_data = {
        'Cases': [40, 20],  # [exposed_cases, unexposed_cases]
        'Non-cases': [60, 80]  # [exposed_noncases, unexposed_noncases]
    }
    contingency_df = pd.DataFrame(default_data, index=['Exposed', 'Unexposed'])

    # Display an editable data table
    edited_df = st.data_editor(
        contingency_df, 
        key="contingency_table",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Cases": st.column_config.NumberColumn(
                "Cases",
                help="Number of individuals with the outcome",
                min_value=0,
                format="%d"
            ),
            "Non-cases": st.column_config.NumberColumn(
                "Non-cases",
                help="Number of individuals without the outcome",
                min_value=0,
                format="%d"
            )
        }
    )

    # Extract the values from the edited table
    exposed_cases = edited_df.iloc[0, 0]  # Exposed Cases (row 0, column 0)
    exposed_noncases = edited_df.iloc[0, 1]  # Exposed Non-cases (row 0, column 1)
    unexposed_cases = edited_df.iloc[1, 0]  # Unexposed Cases (row 1, column 0)
    unexposed_noncases = edited_df.iloc[1, 1]  # Unexposed Non-cases (row 1, column 1)

    # Calculate measures
    measures = calculate_measures(
        exposed_cases, exposed_noncases,
        unexposed_cases, unexposed_noncases
    )

    ########################
    # Display the Results  #
    ########################
    st.header("Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Risk in Exposed", f"{measures['Risk in Exposed']:.2f}")
        st.metric("Risk Ratio (RR)", f"{measures['Risk Ratio']:.2f}")

    with col4:
        st.metric("Risk in Unexposed", f"{measures['Risk in Unexposed']:.2f}")
        st.metric("Odds Ratio (OR)", f"{measures['Odds Ratio']:.2f}")

    st.metric("Risk Difference (RD)", f"{measures['Risk Difference']:.2f}")

    ##############################
    # Visualization (Bar Chart)  #
    ##############################
    data = pd.DataFrame({
        'Exposure': ['Exposed', 'Exposed', 'Unexposed', 'Unexposed'],
        'Outcome': ['Case', 'Non-case', 'Case', 'Non-case'],
        'Count': [exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases]
    })

    fig = px.bar(
        data,
        x='Exposure',
        y='Count',
        color='Outcome',
        barmode='group',
        title='Distribution of Cases and Non-cases by Exposure Status'
    )

    st.plotly_chart(fig, use_container_width=True)

    ############################
    # Educational Explanation  #
    ############################
    st.header("Understanding Measures of Association")

    st.subheader("Risk Ratio (Relative Risk)")
    st.markdown("""
    The **Risk Ratio (RR)** is the ratio of the risk in the exposed group to the risk in the unexposed group.

    - **RR = 1**: No association.
    - **RR > 1**: Exposure associated with increased risk.
    - **RR < 1**: Exposure associated with decreased risk (protective).
    """)

    st.subheader("Odds Ratio")
    st.markdown("""
    The **Odds Ratio (OR)** compares the odds of exposure among cases to the odds of exposure among controls.
    - Often used in case-control studies where risk cannot be directly calculated.
    - Approximates the risk ratio when the disease is rare.
    - Interpreted similarly to a risk ratio (but *not* exactly the same).
    """)

    st.subheader("Risk Difference (Attributable Risk)")
    st.markdown("""
    The **Risk Difference** is the absolute difference in risk between the exposed and unexposed groups.
    - Measures the absolute impact of exposure.
    - Useful for public health impact assessments.
    - Can be used to calculate "number needed to treat/harm." 
    """)

    st.header("When to Use Each Measure")
    st.markdown("""
    1. **Risk Ratio**:
    - Best for cohort studies.
    - Communicates relative effects.

    2. **Odds Ratio**:
    - Best for case-control studies.
    - Particularly suitable when the disease is rare.
    - Often used in logistic regression outputs.

    3. **Risk Difference**:
    - Important for public health planning and cost-benefit analyses.
    - Provides an absolute measure of effect (how many cases prevented or caused).
    """)

    ##########################
    # Interactive Quiz/Check #
    ##########################
    st.subheader("Test Your Understanding")
    quiz_q = st.radio(
        "Which measure is best used in case-control studies, particularly when the disease is rare?",
        ["Risk Ratio (RR)", "Odds Ratio (OR)", "Risk Difference (RD)"]
    )

    if quiz_q == "Odds Ratio (OR)":
        st.success("Correct! The odds ratio is typically used in case-control designs.")
    else:
        st.error("Not quite. In case-control studies, we often can't directly compute risk, so we use the OR.")

with code_tab:
    measure_of_association_code.app()
    