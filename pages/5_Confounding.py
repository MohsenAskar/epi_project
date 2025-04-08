# pages/3_confounding.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy import stats

import sys
import os

# Add path to import the code lab module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interactive_code'))
import confounding_code

############################
# Data Generation Function #
############################
def generate_confounding_data(n_samples=1000, confounder_strength=0.5, true_effect=0.3, scenario="age_smoking_cancer"):
    """
    Generate data with confounding.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points to generate
    confounder_strength : float
        How strongly the confounder influences both exposure and outcome
    true_effect : float
        The actual causal effect of exposure on outcome
    scenario : str
        Which real-world example to simulate
    """
    # Generate confounder
    confounder = np.random.normal(0, 1, n_samples)
    
    # Generate exposure influenced by confounder
    exposure = confounder * confounder_strength + np.random.normal(0, 1, n_samples)
    
    # Generate outcome influenced by both exposure and confounder
    outcome = (
        exposure * true_effect + 
        confounder * confounder_strength + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Create descriptive variable names based on scenario
    if scenario == "age_smoking_cancer":
        confounder_name = "Age (years)"
        exposure_name = "Smoking (pack-years)"
        outcome_name = "Cancer Risk"
    elif scenario == "ses_diet_health":
        confounder_name = "Socioeconomic Status"
        exposure_name = "Diet Quality"
        outcome_name = "Health Outcome"
    elif scenario == "education_exercise_longevity":
        confounder_name = "Education Level"
        exposure_name = "Exercise Frequency"
        outcome_name = "Longevity"
    else:
        confounder_name = "Confounder"
        exposure_name = "Exposure"
        outcome_name = "Outcome"
    
    # Scale variables to match real-world quantities
    if scenario == "age_smoking_cancer":
        confounder = confounder * 15 + 45  # Age centered around 45 years
        exposure = exposure * 10 + 15      # Smoking centered around 15 pack-years
        outcome = outcome * 5 + 10         # Risk score centered around 10
    
    return pd.DataFrame({
        confounder_name: confounder,
        exposure_name: exposure,
        outcome_name: outcome
    })

def calculate_regression_results(data):
    """Calculate crude and adjusted regression coefficients."""
    # Extract column names
    cols = list(data.columns)
    confounder_name, exposure_name, outcome_name = cols[0], cols[1], cols[2]
    
    # Crude model (without adjusting for confounder)
    X_crude = sm.add_constant(data[exposure_name])
    crude_model = sm.OLS(data[outcome_name], X_crude).fit()
    crude_coef = crude_model.params[exposure_name]
    crude_p = crude_model.pvalues[exposure_name]
    crude_r2 = crude_model.rsquared
    
    # Adjusted model (controlling for confounder)
    X_adjusted = sm.add_constant(data[[exposure_name, confounder_name]])
    adjusted_model = sm.OLS(data[outcome_name], X_adjusted).fit()
    adjusted_coef = adjusted_model.params[exposure_name]
    adjusted_p = adjusted_model.pvalues[exposure_name]
    adjusted_r2 = adjusted_model.rsquared
    
    # Calculate correlations
    corr_exposure_outcome = data[exposure_name].corr(data[outcome_name])
    corr_confounder_exposure = data[confounder_name].corr(data[exposure_name])
    corr_confounder_outcome = data[confounder_name].corr(data[outcome_name])
    
    return {
        "crude_coef": crude_coef,
        "crude_p": crude_p,
        "crude_r2": crude_r2,
        "adjusted_coef": adjusted_coef,
        "adjusted_p": adjusted_p,
        "adjusted_r2": adjusted_r2,
        "corr_exposure_outcome": corr_exposure_outcome,
        "corr_confounder_exposure": corr_confounder_exposure,
        "corr_confounder_outcome": corr_confounder_outcome
    }

################
# Page Layout  #
################
st.set_page_config(layout="wide")

# Title Section with clear branding
st.title("🔬 Understanding Confounding in Epidemiology")
st.markdown("### An interactive visualization tool for students")

######################################
# Create columns for controls & plot #
######################################
viz_tab, code_tab = st.tabs(["📊 Interactive Visualization", "💻 Code Laboratory"])

# Wrap your existing visualization content in:
with viz_tab:

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("Simulation Controls")
        
        # Example selector
        scenario = st.selectbox(
            "Choose a Real-World Example:",
            options=["age_smoking_cancer", "ses_diet_health", "education_exercise_longevity"],
            format_func=lambda x: {
                "age_smoking_cancer": "🚬 Age confounding Smoking-Cancer relationship",
                "ses_diet_health": "💰 Socioeconomic Status confounding Diet-Health relationship",
                "education_exercise_longevity": "🎓 Education confounding Exercise-Longevity relationship"
            }[x]
        )

        # Sliders
        confounder_strength = st.slider(
            "Confounder Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="How strongly the confounder influences both the exposure and outcome"
        )

        true_effect = st.slider(
            "True Exposure Effect",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="The actual causal effect of the exposure on the outcome"
        )
        
        # Explanation box with improved formatting
        st.markdown("""
        ### 📊 Simulation Parameters
        
        **Confounder Strength** determines how strongly the third variable 
        influences both exposure and outcome. Higher values create stronger 
        confounding effects.
        
        **True Exposure Effect** is the actual causal influence of exposure
        on the outcome. This represents what we'd measure in a perfect study
        with no confounding.
        
        ### 🔎 What to Look For
        
        1. Compare the crude relationship (top-left) with the true effect
        2. Notice how the stratified analysis (bottom-right) reduces confounding
        3. Check how the estimated effect changes when adjusting for the confounder
        """)
        
        # Toggle for advanced features
        show_dag = st.checkbox("Show Causal Diagram (DAG)", value=True)
        show_metrics = st.checkbox("Show Statistical Metrics", value=True)

    with col2:
        # Generate data
        data = generate_confounding_data(
            n_samples=1000,
            confounder_strength=confounder_strength,
            true_effect=true_effect,
            scenario=scenario
        )
        
        # Extract column names from data
        cols = list(data.columns)
        confounder_name, exposure_name, outcome_name = cols[0], cols[1], cols[2]
        
        # Calculate regression results
        try:
            results = calculate_regression_results(data)
        except Exception as e:
            # Fallback values if regression fails
            results = {
                "crude_coef": confounder_strength + true_effect,
                "crude_p": 0.01,
                "crude_r2": 0.3,
                "adjusted_coef": true_effect,
                "adjusted_p": 0.05,
                "adjusted_r2": 0.2,
                "corr_exposure_outcome": confounder_strength + true_effect,
                "corr_confounder_exposure": confounder_strength,
                "corr_confounder_outcome": confounder_strength
            }
            st.warning(f"Statistical calculation simplified due to an error: {str(e)}")

        # DAG Visualization (if enabled)
        if show_dag:
            st.markdown("### Causal Diagram (DAG)")
            
            # Add NetworkX-style graph visualization using Plotly
            import networkx as nx
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes with better position for confounding (confounder at top)
            G.add_node("confounder", pos=(1, 1), name=confounder_name)
            G.add_node("exposure", pos=(0, 0), name=exposure_name)
            G.add_node("outcome", pos=(2, 0), name=outcome_name)
            
            # Add edges to show proper confounding relationship
            G.add_edge("confounder", "exposure")
            G.add_edge("confounder", "outcome")
            G.add_edge("exposure", "outcome")
            
            # Get positions from NetworkX
            pos = nx.get_node_attributes(G, 'pos')
            
            # Create figure
            dag_fig = go.Figure()
            
            # Add edges as arrows
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                dag_fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines',
                        line=dict(width=2, color='black', dash='solid'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
            
            # Add nodes
            node_colors = {'confounder': '#FFD700', 'exposure': '#90EE90', 'outcome': '#ADD8E6'}
            
            for node in G.nodes():
                x, y = pos[node]
                name = G.nodes[node]['name']
                dag_fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(size=40, color=node_colors[node]),
                        text=name,
                        textposition="top center",
                        hoverinfo='text',
                        hovertext=name,
                        showlegend=False
                    )
                )
                
            # Add arrowheads
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Calculate position for arrowhead (slightly before the endpoint)
                arrow_ratio = 0.9  # Position at 90% of the way
                ax = x0 + arrow_ratio * (x1 - x0)
                ay = y0 + arrow_ratio * (y1 - y0)
                
                dag_fig.add_annotation(
                    x=x1, y=y1,
                    ax=ax, ay=ay,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='black'
                )
                
            # Update layout
            dag_fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-0.5, 2.5]
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-0.5, 1.5]
                ),
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                title="Causal Relationships"
            )
            
            # Add explanatory text
            st.markdown("""
            This diagram shows the correct confounding structure:
            - The confounder affects both the exposure and the outcome
            - The exposure affects the outcome (true causal relationship)
            - No arrow from exposure to confounder (confounder isn't affected by exposure)
            
            ⚠️ Note: In confounding, the confounder is NOT a mediator, but rather a common cause of both exposure and outcome.
            """)
            
            st.plotly_chart(dag_fig, use_container_width=True)
        
        # Statistical Metrics Display (if enabled)
        if show_metrics:
            st.markdown("### Statistical Analysis")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.markdown("**Correlations:**")
                st.metric(f"{exposure_name} vs {outcome_name}", f"{results['corr_exposure_outcome']:.3f}")
                st.metric(f"{confounder_name} vs {exposure_name}", f"{results['corr_confounder_exposure']:.3f}")
                st.metric(f"{confounder_name} vs {outcome_name}", f"{results['corr_confounder_outcome']:.3f}")
            
            with metrics_col2:
                st.markdown("**Regression Coefficients:**")
                crude_color = 'normal' if abs(results['crude_coef'] - true_effect) < 0.1 else 'off'
                adjusted_color = 'normal' if abs(results['adjusted_coef'] - true_effect) < 0.1 else 'off'
                
                st.metric(
                    "Crude Effect (Unadjusted)", 
                    f"{results['crude_coef']:.3f}", 
                    f"{results['crude_coef'] - true_effect:.3f}",
                    delta_color=crude_color
                )
                st.metric(
                    "Adjusted Effect (Controlling for Confounder)", 
                    f"{results['adjusted_coef']:.3f}", 
                    f"{results['adjusted_coef'] - true_effect:.3f}",
                    delta_color=adjusted_color
                )
                st.metric("True Effect (Actual Parameter)", f"{true_effect:.3f}")

    # Create visualizations using subplots
    st.markdown("---") 
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{exposure_name} vs {outcome_name} (Crude)',
            f'{confounder_name} vs {exposure_name}',
            f'{confounder_name} vs {outcome_name}',
            'Stratified Analysis'
        )
    )

    # 1) Crude association: Exposure vs Outcome
    fig.add_trace(
        go.Scatter(
            x=data[exposure_name],
            y=data[outcome_name],
            mode='markers',
            name='Crude',
            marker=dict(size=5, color='blue', opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Add regression line for crude relationship
    x_range = np.linspace(data[exposure_name].min(), data[exposure_name].max(), 100)
    y_pred = results['crude_coef'] * x_range + results['crude_r2'] # Using R² as a proxy for intercept
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name='Crude Regression',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # 2) Confounder vs Exposure
    fig.add_trace(
        go.Scatter(
            x=data[confounder_name],
            y=data[exposure_name],
            mode='markers',
            name='Conf-Exp',
            marker=dict(size=5, color='green', opacity=0.6)
        ),
        row=1, col=2
    )

    # 3) Confounder vs Outcome
    fig.add_trace(
        go.Scatter(
            x=data[confounder_name],
            y=data[outcome_name],
            mode='markers',
            name='Conf-Out',
            marker=dict(size=5, color='purple', opacity=0.6)
        ),
        row=2, col=1
    )

    # 4) Stratified analysis by confounder levels
    strata = pd.qcut(data[confounder_name], q=3, labels=['Low', 'Medium', 'High'])
    colors = ['blue', 'green', 'red']
    
    # Plot stratified data and add regression lines for each stratum
    for stratum, color in zip(strata.unique(), colors):
        mask = strata == stratum
        stratum_data = data[mask]
        
        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=stratum_data[exposure_name],
                y=stratum_data[outcome_name],
                mode='markers',
                name=f'Stratum {stratum}',
                marker=dict(size=5, color=color, opacity=0.6)
            ),
            row=2, col=2
        )
        
        # Calculate and add regression line for this stratum
        if len(stratum_data) > 2:  # Need at least 3 points for regression
            X = sm.add_constant(stratum_data[exposure_name])
            model = sm.OLS(stratum_data[outcome_name], X).fit()
            
            x_range = np.linspace(stratum_data[exposure_name].min(), stratum_data[exposure_name].max(), 100)
            y_pred = model.params[exposure_name] * x_range + model.params['const']
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name=f'Regression {stratum}',
                    line=dict(color=color, width=2)
                ),
                row=2, col=2
            )

    # Update layout for better readability
    fig.update_layout(
        height=700, 
        autosize=True,  # Enable autosize
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=0.9
        ),
        template="simple_white"
    )
    
    # Add annotations in better positions with shorter text
    fig.add_annotation(
        x=0.25, y=0.95, 
        xref="paper", yref="paper",
        text="Unadjusted relationship",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        x=0.75, y=0.95, 
        xref="paper", yref="paper",
        text="Confounder → Exposure",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        x=0.25, y=0.45, 
        xref="paper", yref="paper",
        text="Confounder → Outcome",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        x=0.75, y=0.45, 
        xref="paper", yref="paper",
        text="Stratified by confounder",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update axis labels
    fig.update_xaxes(title_text=exposure_name, row=1, col=1)
    fig.update_yaxes(title_text=outcome_name, row=1, col=1)
    
    fig.update_xaxes(title_text=confounder_name, row=1, col=2)
    fig.update_yaxes(title_text=exposure_name, row=1, col=2)
    
    fig.update_xaxes(title_text=confounder_name, row=2, col=1)
    fig.update_yaxes(title_text=outcome_name, row=2, col=1)
    
    fig.update_xaxes(title_text=exposure_name, row=2, col=2)
    fig.update_yaxes(title_text=outcome_name, row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

    ########################
    # Educational Content  #
    ########################
    st.header("🧠 Understanding Confounding")

    tab1, tab2, tab3 = st.tabs(["Concept", "Examples", "Methods to Control"])

    with tab1:
        st.markdown("""
        ### What is Confounding?
        
        **Confounding** occurs when a third variable (the confounder) influences both the exposure and outcome,
        creating a distorted association between them. It's one of the most important concepts in epidemiology
        and can lead researchers to incorrect conclusions if not properly addressed.
        
        #### Key characteristics of a confounder:
        
        1. Associated with the exposure
        2. An independent risk factor for the outcome
        3. Not on the direct causal pathway between exposure and outcome (not a mediator)
        
        In our simulation, changing the **Confounder Strength** shows how much the relationship between exposure
        and outcome can be distorted by confounding. The difference between the crude and adjusted effects shows
        the amount of confounding.
        """)
        
    with tab2:
        st.markdown("""
        ### Real-World Examples of Confounding
        
        #### 1. Age Confounding the Smoking-Cancer Relationship
        
        - **Exposure**: Smoking (measured as pack-years)
        - **Outcome**: Cancer risk
        - **Confounder**: Age
        
        Both smoking duration and cancer risk increase with age. If we don't adjust for age,
        we might overestimate the effect of smoking on cancer.
        
        #### 2. Socioeconomic Status Confounding Diet-Health Relationship
        
        - **Exposure**: Diet quality
        - **Outcome**: Health outcomes
        - **Confounder**: Socioeconomic status (SES)
        
        People with higher SES often have better diets and better health outcomes for many reasons
        (better healthcare access, less stress, etc.). Without adjusting for SES, we might attribute
        too much of the health benefit to diet alone.
        
        #### 3. Education Confounding Exercise-Longevity Relationship
        
        - **Exposure**: Exercise frequency
        - **Outcome**: Longevity (lifespan)
        - **Confounder**: Education level
        
        More educated people tend to exercise more and also live longer (due to many factors).
        If we don't account for education level, we might overestimate how much exercise itself
        extends lifespan.
        """)

    with tab3:
        st.markdown("""
        ### Methods to Control for Confounding
        
        #### Study Design Methods:
        
        1. **Randomization**: Randomly assigning exposure status eliminates confounding by creating groups that 
        are balanced on all variables (observed and unobserved).
        
        2. **Matching**: Selecting comparison groups with similar distributions of potential confounders.
        
        3. **Restriction**: Limiting the study to individuals with similar values of potential confounders.
        
        #### Statistical Methods:
        
        1. **Stratification**: Analyzing data within subgroups (strata) of the confounder, as shown in the
        bottom-right plot of our visualization.
        
        2. **Multivariable Regression**: Including potential confounders as covariates in regression models.
        This is what we're doing in the "Adjusted Effect" calculation.
        
        3. **Propensity Score Methods**: Creating a score representing the probability of exposure based on
        confounding variables, then adjusting for this score.
        
        4. **Instrumental Variables**: Using a variable that affects the outcome only through the exposure.
        """)

 ###########################
# Interactive Quiz/Checks #
###########################
st.header("🧐 Test Your Understanding")

# More comprehensive quiz with multiple questions
quiz_tab1, quiz_tab2 = st.tabs(["Basic Concepts", "Applied Knowledge"])

with quiz_tab1:
    q1_options = [
        "-- Select an answer --",
        "Because it introduces measurement error in the exposure variable.",
        "Because it randomly changes the outcome without affecting the exposure.",
        "Because the confounder is related to both the exposure and the outcome, creating a spurious association.",
        "Because it only affects people who are susceptible to the outcome."
    ]
    q1 = st.radio(
        "1. Which of the following best describes why confounding can distort the observed relationship?",
        q1_options,
        index=0,
        key="q1"
    )

    if q1 != q1_options[0]:  # Only check if user selected a real option
        if q1 == q1_options[3]:
            st.success("✅ Correct! A confounder is associated with both exposure and outcome, distorting the observed effect.")
        else:
            st.error("❌ Not quite. Remember, the key point is that the confounder is associated with both the exposure and the outcome.")

    q2_options = [
        "-- Select an answer --",
        "It becomes closer to the true effect.",
        "It becomes more distorted from the true effect.",
        "It remains unchanged.",
        "It becomes negative even when the true effect is positive."
    ]
    q2 = st.radio(
        "2. In the simulation, what happens to the crude (unadjusted) effect when you increase the confounder strength?",
        q2_options,
        index=0,
        key="q2"
    )

    if q2 != q2_options[0]:
        if q2 == q2_options[2]:
            st.success("✅ Correct! As confounder strength increases, the crude effect becomes increasingly biased.")
        else:
            st.error("❌ Try again. Watch what happens to the difference between crude and adjusted effects as you increase confounder strength.")

with quiz_tab2:
    q3_options = [
        "-- Select an answer --",
        "Because age reduces the harmful effects of smoking.",
        "Because age is independently associated with both smoking duration and cancer risk.",
        "Because older people smoke less than younger people.",
        "Because the measurement of smoking becomes less accurate with age."
    ]
    q3 = st.radio(
        "3. In the age-smoking-cancer example, why might we see a stronger association between smoking and cancer in the crude analysis compared to after adjusting for age?",
        q3_options,
        index=0,
        key="q3"
    )

    if q3 != q3_options[0]:
        if q3 == q3_options[3]:
            st.success("✅ Correct! Age affects both the exposure (smoking duration increases with age) and the outcome (cancer risk increases with age).")
        else:
            st.error("❌ Consider how age relates to both smoking patterns and cancer risk independently.")

    q4_options = [
        "-- Select an answer --",
        "The regression lines for each stratum have similar slopes to the adjusted effect.",
        "The data points become more scattered and random.",
        "The regression lines cross each other at the center.",
        "The regression lines all have slopes of zero."
    ]
    q4 = st.radio(
        "4. Looking at the stratified analysis (bottom-right plot), what pattern indicates that confounding has been reduced?",
        q4_options,
        index=0,
        key="q4"
    )

    if q4 != q4_options[0]:
        if q4 == q4_options[1]:
            st.success("✅ Correct! When we stratify by the confounder, the slopes within each stratum should be closer to the true effect.")
        else:
            st.error("❌ Think about what the slope represents in each stratum and how it relates to the true effect.")

    #######################
    # Additional Features #
    #######################
    st.header("🔧 Try It Yourself")

    st.markdown("""
    ### Experiment with different scenarios

    Try these experiments to deepen your understanding:

    1. Set the **True Effect** to zero, but keep a strong confounder. Notice how it creates the illusion of an association even when there isn't one!

    2. Set the **Confounder Strength** to zero. See how the crude and adjusted effects become identical when there's no confounding.

    3. Try an extreme case: set **Confounder Strength** to 1.0 and **True Effect** to 0.1. Notice how dramatically the crude effect can be distorted.

    4. Compare the different real-world scenarios and see if confounding operates similarly in each case.
    """)

    st.markdown("""
    ---
    ### References and Further Reading

    - Rothman KJ, Greenland S, Lash TL. (2008). *Modern Epidemiology*. 3rd Edition. Philadelphia: Lippincott Williams & Wilkins.
    - [Confounding in Epidemiology (CDC)](https://www.cdc.gov/csels/dsepd/ss1978/lesson4/section4.html)
    - [Controlling for Confounding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4447039/)
    - Hernán MA, Robins JM. (2020). *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC.
    """)
    
with code_tab:
    confounding_code.app()