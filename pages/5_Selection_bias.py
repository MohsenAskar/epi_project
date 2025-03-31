# pages/5_selection_bias.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

# Add path to import the code lab module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interactive_code'))

import selection_bias_code

###############################
# Data Generation Functions   #
###############################
def generate_selection_bias_data(n_samples=1000, selection_strength=0.5, true_effect=0.3, 
                                bias_type="outcome_dependent", confounding_strength=0.2):
    """
    Generate data demonstrating different types of selection bias.
    
    Parameters:
    -----------
    n_samples : int
        Number of individuals in the simulated population
    selection_strength : float
        Strength of the selection bias (0 = no bias, 1 = maximum bias)
    true_effect : float
        True causal effect of exposure on outcome
    bias_type : str
        Type of selection bias to simulate
    confounding_strength : float
        Strength of confounding (used in some bias types)
    """
    # Initialize DataFrame
    df = pd.DataFrame()
    
    # Generate confounder if needed
    confounder = np.random.normal(0, 1, n_samples)
    
    # Generate exposure with/without confounding
    if bias_type in ["differential_loss", "healthy_worker"]:
        # Exposure influenced by confounder
        p_exposure = 1 / (1 + np.exp(-confounding_strength * confounder))
        exposure = np.random.binomial(1, p_exposure, n_samples)
    else:
        # Simple random exposure
        exposure = np.random.binomial(1, 0.5, n_samples)
    
    # Generate outcome based on exposure and possibly confounder
    if bias_type in ["differential_loss", "healthy_worker"]:
        # Outcome influenced by both exposure and confounder
        logit = -0.8 + true_effect * exposure + confounding_strength * confounder
        p_outcome = 1 / (1 + np.exp(-logit))
    else:
        # Outcome only influenced by exposure
        p_outcome = 0.3 + true_effect * exposure
    
    outcome = np.random.binomial(1, p_outcome, n_samples)
    
    # Generate selection probability based on bias type
    if bias_type == "outcome_dependent":
        # Classic example: People with disease more likely to participate
        selection_prob = 0.5 + selection_strength * outcome
    
    elif bias_type == "exposure_dependent":
        # Example: Exposed individuals more likely to participate
        selection_prob = 0.5 + selection_strength * exposure
    
    elif bias_type == "differential_loss":
        # Example: Follow-up loss related to both exposure and outcome
        selection_prob = 0.9 - selection_strength * (outcome * exposure)
    
    elif bias_type == "healthy_worker":
        # Healthy worker effect: Healthier people more likely to be employed (exposed)
        # and less likely to have negative outcomes
        selection_prob = 0.5 + selection_strength * (exposure * (1 - outcome))
    
    elif bias_type == "berkson":
        # Berkson's bias: Both exposure and outcome increase probability of selection
        # (e.g., hospital admission)
        selection_prob = 0.3 + selection_strength * (exposure + outcome) / 2
    
    else:
        # Default: random selection (no bias)
        selection_prob = 0.5 * np.ones(n_samples)
    
    # Ensure probability is between 0 and 1
    selection_prob = np.clip(selection_prob, 0, 1)
    
    # Generate selection based on probability
    selected = np.random.binomial(1, selection_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Confounder': confounder,
        'Exposure': exposure,
        'Outcome': outcome,
        'Selection_Probability': selection_prob,
        'Selected': selected
    })
    
    return df

def calculate_measures_of_association(data, selected_only=False):
    """
    Calculate various measures of association between exposure and outcome.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing Exposure, Outcome, and Selected columns
    selected_only : bool
        If True, calculate only for selected individuals
    
    Returns:
    --------
    dict
        Dictionary of calculated measures
    """
    # Filter data if needed
    if selected_only:
        data = data[data['Selected'] == 1]
    
    # Return empty results if no data (edge case)
    if len(data) < 10:
        return {
            'risk_ratio': None,
            'odds_ratio': None,
            'risk_difference': None,
            'sample_size': 0,
            'exposure_prevalence': None,
            'outcome_prevalence': None
        }
    
    # Create contingency table
    contingency = pd.crosstab(data['Exposure'], data['Outcome'])
    
    # Check if all cells have values; if not, add 0.5 to all cells (continuity correction)
    if 0 in contingency.index and 1 in contingency.index and 0 in contingency.columns and 1 in contingency.columns:
        a = contingency.loc[1, 1]  # Exposed with outcome
        b = contingency.loc[1, 0]  # Exposed without outcome
        c = contingency.loc[0, 1]  # Unexposed with outcome
        d = contingency.loc[0, 0]  # Unexposed without outcome
    else:
        # Add 0.5 to all cells if any are missing
        a = contingency.loc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0.5
        b = contingency.loc[1, 0] if 1 in contingency.index and 0 in contingency.columns else 0.5
        c = contingency.loc[0, 1] if 0 in contingency.index and 1 in contingency.columns else 0.5
        d = contingency.loc[0, 0] if 0 in contingency.index and 0 in contingency.columns else 0.5
    
    # Calculate risk in exposed and unexposed
    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)
    
    # Calculate odds in exposed and unexposed
    odds_exposed = a / b if b > 0 else float('inf')
    odds_unexposed = c / d if d > 0 else float('inf')
    
    # Calculate measures of association
    risk_ratio = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
    odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else float('inf')
    risk_difference = risk_exposed - risk_unexposed
    
    # Calculate prevalences
    exposure_prevalence = (a + b) / (a + b + c + d)
    outcome_prevalence = (a + c) / (a + b + c + d)
    
    return {
        'risk_ratio': risk_ratio,
        'odds_ratio': odds_ratio,
        'risk_difference': risk_difference,
        'sample_size': len(data),
        'exposure_prevalence': exposure_prevalence,
        'outcome_prevalence': outcome_prevalence
    }

#################
# Layout Config #
#################
st.set_page_config(layout="wide")

###################
# Title and Intro #
###################
st.title("🔍 Understanding Selection Bias in Epidemiology")
st.markdown("""
This interactive module helps you understand how selection bias can distort the true relationship
between exposure and outcome in epidemiological studies. You can explore different types of
selection bias and see how they affect measures of association.
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Interactive Visualization", "📝 Educational Content", "🧠 Quiz & Exercises", "💻Code Laboratoy"])

with tab1:
    st.header("Selection Bias Simulator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Controls")
        
        # Type of selection bias
        bias_type = st.selectbox(
            "Type of Selection Bias",
            options=["outcome_dependent", "exposure_dependent", "differential_loss", "healthy_worker", "berkson"],
            format_func=lambda x: {
                "outcome_dependent": "Outcome-Dependent Selection",
                "exposure_dependent": "Exposure-Dependent Selection", 
                "differential_loss": "Differential Loss to Follow-up",
                "healthy_worker": "Healthy Worker Effect",
                "berkson": "Berkson's Bias"
            }[x]
        )
        
        # Bias strength
        selection_strength = st.slider(
            "Selection Bias Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="How strongly selection is related to exposure and/or outcome"
        )
        
        # True effect size
        true_effect = st.slider(
            "True Effect Size",
            min_value=0.0,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="The true causal effect of exposure on outcome"
        )
        
        # Additional parameters in an expander
        with st.expander("Advanced Parameters"):
            confounding_strength = st.slider(
                "Confounding Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Strength of confounding (only relevant for some bias types)"
            )
            
            n_samples = st.slider(
                "Population Size",
                min_value=500,
                max_value=5000,
                value=1000,
                step=500
            )
        
        # Generate button
        if st.button("Generate New Data"):
            st.session_state.refresh_data = True
        else:
            if 'refresh_data' not in st.session_state:
                st.session_state.refresh_data = False
        
        # Bias type explanation
        st.subheader("About This Bias Type")
        
        if bias_type == "outcome_dependent":
            st.markdown("""
            **Outcome-Dependent Selection** occurs when individuals with the outcome are more likely to be included in the study.
            
            **Example:** A study of a disease where symptomatic individuals (outcome=1) are more likely to participate or be diagnosed.
            
            **Impact:** May inflate the apparent prevalence of the outcome and distort measures of association.
            """)
        
        elif bias_type == "exposure_dependent":
            st.markdown("""
            **Exposure-Dependent Selection** occurs when exposed individuals are more likely to be included in the study.
            
            **Example:** Occupational studies where only current workers (exposed) are included, 
            while former workers who left due to health issues are excluded.
            
            **Impact:** May create artificial associations or mask true effects.
            """)
        
        elif bias_type == "differential_loss":
            st.markdown("""
            **Differential Loss to Follow-up** occurs when participants drop out of a study in a way related to both 
            exposure and outcome.
            
            **Example:** In a drug trial, patients experiencing side effects (exposed with outcome) are more likely to drop out.
            
            **Impact:** Usually biases results toward the null (underestimates the true effect).
            """)
        
        elif bias_type == "healthy_worker":
            st.markdown("""
            **Healthy Worker Effect** is a specific form of selection bias in occupational studies.
            
            **Example:** People who are employed (exposed) tend to be healthier than the general population, 
            leading to lower mortality/morbidity rates among workers.
            
            **Impact:** Typically underestimates occupational health risks.
            """)
        
        elif bias_type == "berkson":
            st.markdown("""
            **Berkson's Bias** occurs in hospital-based studies where both exposure and outcome influence 
            the probability of being included.
            
            **Example:** A study conducted among hospitalized patients, where both the exposure and the outcome 
            increase likelihood of hospitalization.
            
            **Impact:** Can create spurious associations between unrelated variables.
            """)
    
    with col2:
        # Generate data
        data = generate_selection_bias_data(
            n_samples=n_samples,
            selection_strength=selection_strength,
            true_effect=true_effect,
            bias_type=bias_type,
            confounding_strength=confounding_strength
        )
        
        # Calculate measures of association for both true population and selected sample
        true_measures = calculate_measures_of_association(data, selected_only=False)
        selected_measures = calculate_measures_of_association(data, selected_only=True)
        
        # Display sample size information
        st.subheader("Sample Information")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric(
                "Total Population", 
                f"{len(data):,}",
                help="Number of individuals in the total population"
            )
        
        with col_b:
            selection_rate = len(data[data['Selected'] == 1]) / len(data) * 100
            st.metric(
                "Selected Sample", 
                f"{len(data[data['Selected'] == 1]):,}",
                f"{selection_rate:.1f}% of population",
                help="Number of individuals selected into the study"
            )
        
        with col_c:
            relative_difference = ((selected_measures['risk_ratio'] / true_measures['risk_ratio']) - 1) * 100 if true_measures['risk_ratio'] and selected_measures['risk_ratio'] else float('nan')
            st.metric(
                "Bias in Risk Ratio", 
                f"{relative_difference:+.1f}%",
                delta_color="inverse",
                help="Relative difference between measured and true risk ratio"
            )
        
        # Create visualizations
        st.subheader("Visualization of Selection Bias")
        
        # Visualization type selector
        viz_type = st.radio(
            "Select Visualization",
            options=["Selection Mechanism", "Measures of Association", "Selection Probabilities"],
            horizontal=True
        )
        
        if viz_type == "Selection Mechanism":
            # Create scatter plot showing who gets selected
            fig = px.scatter(
                data, 
                x="Exposure", 
                y="Outcome",
                color="Selected", 
                color_discrete_map={0: "lightgrey", 1: "red"},
                title="Selection Patterns by Exposure and Outcome Status",
                labels={"Exposure": "Exposure (1=Yes, 0=No)", "Outcome": "Outcome (1=Yes, 0=No)"},
                height=500
            )
            
            # Add jitter directly to the data for better visualization of binary data
            # Instead of using a non-existent 'jitter' property
            jittered_data = data.copy()
            jittered_data['Exposure'] = jittered_data['Exposure'] + np.random.uniform(-0.1, 0.1, len(jittered_data))
            jittered_data['Outcome'] = jittered_data['Outcome'] + np.random.uniform(-0.1, 0.1, len(jittered_data))
            
            # Create a new scatter plot with jittered data
            fig = px.scatter(
                jittered_data, 
                x="Exposure", 
                y="Outcome",
                color="Selected", 
                color_discrete_map={0: "lightgrey", 1: "red"},
                title="Selection Patterns by Exposure and Outcome Status",
                labels={"Exposure": "Exposure (1=Yes, 0=No)", "Outcome": "Outcome (1=Yes, 0=No)"},
                height=500
            )
            
            # Set marker properties
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            
            fig.update_layout(
                xaxis=dict(tickmode='array', tickvals=[0, 1]),
                yaxis=dict(tickmode='array', tickvals=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create selection heatmap
            selection_table = pd.crosstab(
                [data['Exposure'], data['Outcome']], 
                data['Selected'],
                normalize='index'
            ).reset_index()
            
            selection_table = selection_table.rename(columns={
                0: "Percent Not Selected",
                1: "Percent Selected",
                "Exposure": "Exposure Status",
                "Outcome": "Outcome Status"
            })
            
            # Format values as percentages
            selection_table["Percent Selected"] = (selection_table["Percent Selected"] * 100).round(1)
            
            # Create a heatmap of selection percentages
            fig2 = px.imshow(
                pd.pivot_table(
                    selection_table,
                    values="Percent Selected",
                    index="Exposure Status",
                    columns="Outcome Status"
                ),
                text_auto=True,
                labels=dict(x="Outcome Status", y="Exposure Status", color="Percent Selected"),
                x=["No Outcome (0)", "Has Outcome (1)"],
                y=["Not Exposed (0)", "Exposed (1)"],
                color_continuous_scale="Reds",
                title="Percentage Selected in Each Exposure-Outcome Group"
            )
            
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        elif viz_type == "Measures of Association":
            # Create bar chart comparing measures in true vs selected populations
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Risk Ratio", "Odds Ratio", "Risk Difference"],
                shared_yaxes=True
            )
            
            # Add risk ratio bars
            fig.add_trace(
                go.Bar(
                    x=["True Population"],
                    y=[true_measures["risk_ratio"]],
                    name="True Population",
                    marker_color="blue"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=["Selected Sample"],
                    y=[selected_measures["risk_ratio"]],
                    name="Selected Sample",
                    marker_color="red"
                ),
                row=1, col=1
            )
            
            # Add odds ratio bars
            fig.add_trace(
                go.Bar(
                    x=["True Population"],
                    y=[true_measures["odds_ratio"]],
                    name="True Population",
                    marker_color="blue",
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=["Selected Sample"],
                    y=[selected_measures["odds_ratio"]],
                    name="Selected Sample",
                    marker_color="red",
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Add risk difference bars
            fig.add_trace(
                go.Bar(
                    x=["True Population"],
                    y=[true_measures["risk_difference"]],
                    name="True Population",
                    marker_color="blue",
                    showlegend=False
                ),
                row=1, col=3
            )
            
            fig.add_trace(
                go.Bar(
                    x=["Selected Sample"],
                    y=[selected_measures["risk_difference"]],
                    name="Selected Sample",
                    marker_color="red",
                    showlegend=False
                ),
                row=1, col=3
            )
            
            # Add a horizontal line at 1 for ratio measures and 0 for difference
            fig.add_shape(
                type="line",
                x0=-0.5, x1=1.5, y0=1, y1=1,
                line=dict(color="black", width=1, dash="dash"),
                row=1, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=-0.5, x1=1.5, y0=1, y1=1,
                line=dict(color="black", width=1, dash="dash"),
                row=1, col=2
            )
            
            fig.add_shape(
                type="line",
                x0=-0.5, x1=1.5, y0=0, y1=0,
                line=dict(color="black", width=1, dash="dash"),
                row=1, col=3
            )
            
            fig.update_layout(
                title="Impact of Selection Bias on Measures of Association",
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display numeric comparison
            st.subheader("Detailed Comparison")
            
            measures_df = pd.DataFrame({
                "Measure": ["Risk Ratio", "Odds Ratio", "Risk Difference", 
                           "Sample Size", "Exposure Prevalence", "Outcome Prevalence"],
                "True Population": [
                    f"{true_measures['risk_ratio']:.2f}",
                    f"{true_measures['odds_ratio']:.2f}",
                    f"{true_measures['risk_difference']:.2f}",
                    f"{true_measures['sample_size']:,}",
                    f"{true_measures['exposure_prevalence']:.1%}",
                    f"{true_measures['outcome_prevalence']:.1%}"
                ],
                "Selected Sample": [
                    f"{selected_measures['risk_ratio']:.2f}",
                    f"{selected_measures['odds_ratio']:.2f}",
                    f"{selected_measures['risk_difference']:.2f}",
                    f"{selected_measures['sample_size']:,}",
                    f"{selected_measures['exposure_prevalence']:.1%}",
                    f"{selected_measures['outcome_prevalence']:.1%}"
                ]
            })
            
            st.dataframe(
                measures_df, 
                use_container_width=True,
                hide_index=True
            )
        
        else:  # Selection Probabilities
            # Create histogram of selection probabilities
            fig = px.histogram(
                data, 
                x="Selection_Probability", 
                color="Outcome",
                barmode="overlay",
                nbins=20,
                opacity=0.7,
                title="Distribution of Selection Probabilities by Outcome Status",
                labels={"Selection_Probability": "Probability of Being Selected", "Outcome": "Outcome Status"},
                color_discrete_map={0: "blue", 1: "red"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create violin plot showing selection probability by exposure-outcome groups
            fig2 = px.violin(
                data,
                x="Exposure",
                y="Selection_Probability",
                color="Outcome",
                box=True,
                points="all",
                title="Selection Probability by Exposure and Outcome Status",
                labels={"Exposure": "Exposure Status", "Selection_Probability": "Probability of Being Selected", "Outcome": "Outcome Status"},
                color_discrete_map={0: "blue", 1: "red"}
            )
            
            fig2.update_layout(
                xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Not Exposed", "Exposed"])
            )
            
            st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("📚 Understanding Selection Bias")
    
    st.markdown("""
    ### What is Selection Bias?
    
    **Selection bias** occurs when the relationship between exposure and outcome is different in the 
    study population compared to the target population due to non-random selection of participants.
    
    In other words, selection bias happens when the individuals included in a study systematically 
    differ from those who were not included, **in a way that affects the exposure-outcome relationship**.
    
    ### Key Characteristics:
    
    1. The selection process is related to **both** the exposure and the outcome (directly or indirectly)
    2. It distorts the true association between exposure and outcome
    3. It cannot be corrected by simply increasing sample size
    4. It is a systematic error (not random)
    """)
    
    st.subheader("Common Types of Selection Bias")
    
    # Create columns for different bias types
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 1. Self-Selection Bias (Volunteer Bias)
        
        **Definition**: Occurs when individuals who volunteer to participate in a study differ systematically from those who don't.
        
        **Example**: A survey on exercise habits where health-conscious people are more likely to participate.
        
        **DAG Representation**:
        ```
        Exposure → Outcome
             ↓       ↓
             → Selection ←
        ```
        
        ---
        
        #### 2. Loss to Follow-up (Attrition Bias)
        
        **Definition**: Occurs when participants drop out of a study non-randomly.
        
        **Example**: In a study of a medication's side effects, those experiencing side effects (exposed with outcome) are more likely to drop out.
        
        **DAG Representation**:
        ```
        Exposure → Outcome → Loss to Follow-up
                ↗
        ```
        
        ---
        
        #### 3. Healthy Worker Effect
        
        **Definition**: A specific form of selection bias in occupational studies where employed individuals tend to be healthier than the general population.
        
        **Example**: A study comparing disease rates in industrial workers versus the general population might underestimate the health risks of industrial work.
        
        **DAG Representation**:
        ```
        Occupation → Health Outcome
             ↑           ↓
           Health → Employment Status
        ```
        """)
    
    with col2:
        st.markdown("""
        #### 4. Berkson's Bias
        
        **Definition**: Occurs in hospital-based studies where both the exposure and outcome influence the probability of hospital admission.
        
        **Example**: A hospital study finds a spurious association between two diseases because people with either disease are more likely to be hospitalized.
        
        **DAG Representation**:
        ```
        Disease A       Disease B
             ↘           ↙
              Hospitalization
        ```
        
        ---
        
        #### 5. Prevalence-Incidence Bias (Neyman Bias)
        
        **Definition**: Results from studying prevalent (existing) cases rather than incident (new) cases, especially when the exposure affects disease duration.
        
        **Example**: A cross-sectional study of a fatal disease might miss the association with an exposure that increases disease severity and shortens survival.
        
        **DAG Representation**:
        ```
        Exposure → Disease Onset
             ↓         ↓
        Disease Severity → Survival → Inclusion in Study
        ```
        
        ---
        
        #### 6. Non-response Bias
        
        **Definition**: Occurs when non-respondents differ systematically from respondents.
        
        **Example**: A telephone survey where working individuals are less likely to be reached and may have different health behaviors.
        
        **DAG Representation**:
        ```
        Exposure     Outcome
             ↓         ↓
        Factors affecting response → Response to Study
        ```
        """)
    
    st.subheader("How Does Selection Bias Affect Study Results?")
    
    st.markdown("""
    Selection bias can distort the observed association between exposure and outcome in several ways:
    
    1. **Create spurious associations**: When there is no true association, selection bias can create an apparent one
    
    2. **Mask true associations**: Hide a real effect by biasing the result toward the null
    
    3. **Reverse the direction of association**: Make protective factors appear harmful or vice versa
    
    4. **Alter the magnitude of association**: Exaggerate or attenuate the true effect size
    
    The specific impact depends on:
    - The type of selection bias
    - The direction of the selection processes
    - The true relationship between exposure and outcome
    - The measure of association being used (e.g., risk ratio, odds ratio)
    """)
    
    st.subheader("Methods to Prevent and Address Selection Bias")
    
    st.markdown("""
    ### Prevention Strategies:
    
    1. **In Study Design**:
       - Use random sampling from the target population
       - Implement strategies to maximize response rates
       - Follow up with non-respondents to understand differences
       - Use active surveillance systems for outcome ascertainment
    
    2. **In Cohort Studies**:
       - Minimize loss to follow-up
       - Collect reasons for dropout
       - Implement multiple follow-up methods
    
    3. **In Case-Control Studies**:
       - Select controls from the same population that gave rise to the cases
       - Use population-based rather than hospital-based controls when possible
    
    ### Analytical Methods to Address Selection Bias:
    
    1. **Inverse Probability Weighting (IPW)**:
       - Weight observations by the inverse of their selection probability
       - Requires modeling the selection process
    
    2. **Multiple Imputation**:
       - For missing data due to loss to follow-up or non-response
    
    3. **Sensitivity Analysis**:
       - Quantitative bias analysis to assess the potential impact of selection bias
       - Explore different assumptions about the selection mechanism
    
    4. **Adjustment for Selection Variables**:
       - Condition on variables that affected selection (if known and measured)
    
    5. **Heckman Correction**:
       - Two-stage procedure that addresses sample selection bias
    """)
    
    st.subheader("Real-World Examples of Selection Bias")
    
    example1, example2, example3 = st.tabs(["Example 1: COVID-19 Case Reporting", "Example 2: Occupational Studies", "Example 3: Clinical Trials"])
    
    with example1:
        st.markdown("""
        #### Selection Bias in COVID-19 Case Reporting
        
        **Scenario**: Early in the COVID-19 pandemic, testing was often limited to those with symptoms or known exposures.
        
        **Selection Mechanism**: People with symptoms were more likely to get tested and become "cases."
        
        **Bias Impact**: 
        - Overestimation of the case fatality rate (more severe cases were preferentially detected)
        - Underestimation of the true infection prevalence
        - Distorted understanding of risk factors (factors associated with testing were conflated with factors associated with infection)
        
        **Correction Approaches**:
        - Population-based random sampling studies
        - Seroprevalence surveys
        - Adjustment for testing rates in analyses
        """)
    
    with example2:
        st.markdown("""
        #### Healthy Worker Effect in Occupational Studies
        
        **Scenario**: A study comparing mortality rates between industrial workers and the general population finds lower mortality among workers.
        
        **Selection Mechanism**: To be employed, workers typically need to be healthy enough to work ("healthy hire effect"). Those who develop health problems may leave employment ("healthy worker survivor effect").
        
        **Bias Impact**:
        - Underestimation of occupational health risks
        - False conclusion that the industrial work is protective
        
        **Correction Approaches**:
        - Use former workers or workers from different industries as the comparison group
        - Adjust for time since hire and employment status
        - Use internal comparisons (e.g., different exposure levels within the workforce)
        - G-methods to address time-varying confounding affected by prior exposure
        """)
    
    with example3:
        st.markdown("""
        #### Selection Bias in Clinical Trial Dropouts
        
        **Scenario**: In a randomized controlled trial of a new medication, patients experiencing side effects are more likely to drop out.
        
        **Selection Mechanism**: Patients who experience adverse effects (exposed with outcome) leave the study, resulting in fewer documented adverse events.
        
        **Bias Impact**:
        - Underestimation of side effect rates
        - Potentially favorable efficacy estimates if side effects are correlated with treatment response
        
        **Correction Approaches**:
        - Intention-to-treat analysis
        - Inverse probability weighting to account for differential dropout
        - Multiple imputation for missing outcome data
        - Sensitivity analyses with worst-case scenarios for dropouts
        """)

with tab3:
    st.header("Test Your Understanding")
    
    # Multiple quiz questions with increasing difficulty
    st.subheader("Basic Concepts")
    
    q1 = st.radio(
        "1. Which statement correctly describes selection bias?",
        [
            "It occurs when the exposure is measured incorrectly",
            "It occurs when those in the study differ from those not in the study, in a way related to both exposure and outcome",
            "It's a random error that affects all epidemiological studies equally",
            "It happens when the outcome is defined inconsistently across study groups"
        ],
        key="q1"
    )
    
    if q1 == "It occurs when those in the study differ from those not in the study, in a way related to both exposure and outcome":
        st.success("✅ Correct! Selection bias involves non-random selection related to both exposure and outcome.")
    elif q1:  # Only show feedback if an answer is selected
        st.error("❌ That's not right. Selection bias is about who gets into your study and how that relates to both the exposure and outcome.")
    
    q2 = st.radio(
        "2. In the simulation, what happens when you increase the selection bias strength to maximum?",
        [
            "The measures of association in the selected sample become exactly the same as in the true population",
            "The selected sample becomes more representative of the true population",
            "The measures of association in the selected sample become more distorted compared to the true population",
            "Random error increases but systematic bias decreases"
        ],
        key="q2"
    )
    
    if q2 == "The measures of association in the selected sample become more distorted compared to the true population":
        st.success("✅ Correct! Higher selection bias strength leads to greater distortion of the measures of association.")
    elif q2:
        st.error("❌ Not quite. Try experimenting with the slider and observe what happens to the measures.")
    
    st.subheader("Applied Knowledge")
    
    q3 = st.radio(
        "3. A cohort study of a new diet finds that it reduces the risk of heart disease. However, 40% of participants were lost to follow-up, with higher dropout rates among those reporting difficulty following the diet. What type of selection bias might this introduce?",
        [
            "Berkson's bias",
            "Healthy worker effect",
            "Differential loss to follow-up",
            "Prevalence-incidence bias"
        ],
        key="q3"
    )
    
    if q3 == "Differential loss to follow-up":
        st.success("✅ Correct! This is a classic example of differential loss to follow-up, where participants drop out of the study non-randomly.")
    elif q3:
        st.error("❌ Think about what happens when participants drop out of a study non-randomly.")
    
    q4 = st.radio(
        "4. A case-control study uses hospital controls to study risk factors for a rare disease. How might Berkson's bias affect this study?",
        [
            "It would have no effect since case-control studies are immune to selection bias",
            "It could create a spurious association if the exposure increases hospitalization risk independently of the disease",
            "It would always bias results toward showing a protective effect of the exposure",
            "It only affects the precision of the estimates, not their validity"
        ],
        key="q4"
    )
    
    if q4 == "It could create a spurious association if the exposure increases hospitalization risk independently of the disease":
        st.success("✅ Correct! This is exactly how Berkson's bias works - when both the exposure and outcome affect selection (hospitalization), it can create spurious associations.")
    elif q4:
        st.error("❌ Think about what happens when both the exposure and the outcome independently increase the chance of being in the hospital.")
    
    st.subheader("Interactive Exercise: Detecting Selection Bias")
    
    st.markdown("""
    ### Case Study Analysis
    
    **Study Description**: Researchers conducted a case-control study to investigate the relationship between coffee consumption (exposure) and pancreatic cancer (outcome). They recruited cases from a specialty cancer hospital and controls from the general hospital where the specialty hospital is located. The study found that coffee drinkers had 2.5 times the risk of pancreatic cancer compared to non-coffee drinkers.
    
    **Additional Information**:
    - The specialty cancer hospital is a referral center that receives patients from across the country
    - Coffee consumption is associated with smoking
    - Smoking is a known risk factor for many conditions requiring hospitalization, including pancreatic cancer
    """)
    
    options = st.multiselect(
        "Which potential sources of selection bias could affect this study? (Select all that apply)",
        [
            "Berkson's bias, as both coffee consumption (via associated smoking) and pancreatic cancer increase hospitalization probability",
            "Self-selection bias, as coffee drinkers might be more likely to participate in the study",
            "Referral bias, as cancer patients at a specialty hospital may differ from the general population of cancer patients",
            "Healthy worker effect, as employed people drink more coffee and get less cancer",
            "Prevalence-incidence bias, as coffee might affect cancer survival rather than incidence",
            "No selection bias is present in this design"
        ],
        key="case_study"
    )
    
    correct_options = [
        "Berkson's bias, as both coffee consumption (via associated smoking) and pancreatic cancer increase hospitalization probability",
        "Referral bias, as cancer patients at a specialty hospital may differ from the general population of cancer patients"
    ]
    
    if options:  # Only evaluate if options have been selected
        if set(options) == set(correct_options):
            st.success("""
            ✅ Correct! This study is vulnerable to:
            
            1. **Berkson's bias**: Coffee consumption is associated with smoking, which increases hospitalization risk for many conditions. This means coffee drinkers might be overrepresented in the general hospital controls, potentially biasing the association.
            
            2. **Referral bias**: Using cases from a specialty referral center might select for particular types of cancer cases that aren't representative of all pancreatic cancer cases.
            """)
        elif any(option in correct_options for option in options) and all(option in correct_options for option in options):
            st.warning("⚠️ You identified some correct issues, but didn't select all relevant sources of bias.")
        else:
            st.error("❌ Review your selections. Consider how the control selection process and the referral patterns might introduce bias.")
    
    st.subheader("Analytical Correction Exercise")
    
    st.markdown("""
    ### Addressing Selection Bias
    
    **Scenario**: In a cohort study of a drug's effect on heart disease, participants who experienced side effects were more likely to drop out. The crude analysis shows the drug reduces heart disease risk by 50%, but you suspect selection bias.
    
    **Available Information**:
    - 30% of the drug group and 10% of the placebo group dropped out
    - Data on side effects was collected before participants dropped out
    - Pre-existing risk factors for heart disease were measured at baseline
    """)
    
    correction_methods = st.multiselect(
        "Which methods would be appropriate to address this selection bias? (Select all that apply)",
        [
            "Ignore dropouts and analyze only participants who completed the study (per-protocol analysis)",
            "Use inverse probability weighting based on dropout predictors",
            "Conduct sensitivity analyses with multiple imputation for missing outcomes",
            "Use instrumental variable analysis",
            "Adjust for side effects in the analysis",
            "Perform intention-to-treat analysis using last observation carried forward"
        ],
        key="correction"
    )
    
    appropriate_methods = [
        "Use inverse probability weighting based on dropout predictors",
        "Conduct sensitivity analyses with multiple imputation for missing outcomes",
        "Perform intention-to-treat analysis using last observation carried forward"
    ]
    
    if correction_methods:  # Only evaluate if methods have been selected
        if all(method in appropriate_methods for method in correction_methods) and any(method in appropriate_methods for method in correction_methods):
            st.success("""
            ✅ Good choices! These methods can help address the selection bias from differential dropout:
            
            - **Inverse probability weighting** can adjust for the differential dropout by giving more weight to participants who remained in the study but were similar to those who dropped out
            
            - **Multiple imputation** can fill in missing outcome data based on observed characteristics
            
            - **Intention-to-treat analysis** analyzes all randomized participants regardless of whether they completed the study
            """)
        else:
            st.error("""
            ❌ Some of your selections aren't ideal for this scenario:
            
            - Per-protocol analysis would exacerbate the selection bias by completely excluding dropouts
            
            - Adjusting for side effects could create collider bias
            
            - Instrumental variable analysis is typically used for confounding rather than selection bias in this context
            """)

with tab4:
    selection_bias_code.app()
    
# Footer with references
st.markdown("""
---
### References

1. Hernán MA, Hernández-Díaz S, Robins JM. A structural approach to selection bias. Epidemiology. 2004;15(5):615-625.
2. Rothman KJ, Greenland S, Lash TL. Modern Epidemiology. 3rd Edition. Philadelphia: Lippincott Williams & Wilkins; 2008.
3. Griffith GJ, Morris TT, Tudball MJ, et al. Collider bias undermines our understanding of COVID-19 disease risk and severity. Nature Communications. 2020;11(1):5749.
4. Howe CJ, Cole SR, Lau B, Napravnik S, Eron JJ Jr. Selection bias due to loss to follow up in cohort studies. Epidemiology. 2016;27(1):91-97.
5. Westreich D. Berkson's bias, selection bias, and missing data. Epidemiology. 2012;23(1):159-164.

---
*Created for educational purposes to illustrate selection bias concepts in epidemiology.*
""")