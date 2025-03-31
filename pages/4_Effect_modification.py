# pages/4_effect_modification.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import sys
import os

# Add path to import the code lab module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interactive_code'))
import effect_modification_code
#############################
# Data Generation Function  #
#############################
def generate_effect_modification_data(n_samples=1000, effect_modifier_strength=0.5, 
                                     group1_effect=0.2, group2_effect=0.5, group3_effect=0.8,
                                     scenario="age_medication"):
    """Generate data with effect modification.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    effect_modifier_strength : float
        Overall strength of the effect modification
    group1_effect, group2_effect, group3_effect : float
        Group-specific effect sizes
    scenario : str
        Which real-world scenario to simulate
    """
    # Define group labels based on scenario
    if scenario == "age_medication":
        groups = ['Young', 'Middle', 'Old']
    elif scenario == "sex_exercise":
        groups = ['Male', 'Female']  # Biological sex categories
    elif scenario == "genotype_treatment":
        groups = ['AA', 'AB', 'BB']
    else:
        groups = ['Group 1', 'Group 2', 'Group 3']
    
    # Generate effect modifier based on scenario
    effect_modifier = np.random.choice(groups, n_samples)

    # Generate exposure (treatment/intervention)
    exposure = np.random.normal(0, 1, n_samples)

    # Generate outcome with different effects by group
    outcome = np.zeros(n_samples)

    # Apply different effects based on group
    for i, group in enumerate(groups):
        mask = (effect_modifier == group)
        
        # Get the right effect size for this group
        if i == 0:
            effect = group1_effect
        elif i == 1:
            effect = group2_effect
        else:
            effect = group3_effect

        # Generate outcome with group-specific effect
        outcome[mask] = (
            exposure[mask] * effect * effect_modifier_strength +
            np.random.normal(0, 0.5, sum(mask))
        )
    
    # Customize variable names based on scenario
    if scenario == "age_medication":
        modifier_name = "Age Group"
        exposure_name = "Medication Dose"
        outcome_name = "Treatment Effect"
    elif scenario == "sex_exercise":
        modifier_name = "Sex"
        exposure_name = "Exercise Intensity"
        outcome_name = "Weight Loss" 
    elif scenario == "genotype_treatment":
        modifier_name = "Genotype"
        exposure_name = "Drug Dose"
        outcome_name = "Symptom Reduction"
    else:
        modifier_name = "Effect Modifier"
        exposure_name = "Exposure"
        outcome_name = "Outcome"
    
    # Create dataframe with appropriate column names
    df = pd.DataFrame({
        modifier_name: effect_modifier,
        exposure_name: exposure,
        outcome_name: outcome
    })
    
    return df

def calculate_group_statistics(data):
    """Calculate regression coefficients and statistics by group"""
    # Extract column names
    cols = list(data.columns)
    modifier_name, exposure_name, outcome_name = cols[0], cols[1], cols[2]
    
    results = {}
    
    # Overall model
    X = sm.add_constant(data[exposure_name])
    overall_model = sm.OLS(data[outcome_name], X).fit()
    results['overall'] = {
        'coefficient': overall_model.params[exposure_name],
        'p_value': overall_model.pvalues[exposure_name],
        'confidence_interval': overall_model.conf_int().loc[exposure_name].tolist(),
        'r_squared': overall_model.rsquared
    }
    
    # Group-specific models
    for group in data[modifier_name].unique():
        group_data = data[data[modifier_name] == group]
        if len(group_data) > 0:
            X = sm.add_constant(group_data[exposure_name])
            group_model = sm.OLS(group_data[outcome_name], X).fit()
            results[group] = {
                'coefficient': group_model.params[exposure_name],
                'p_value': group_model.pvalues[exposure_name],
                'confidence_interval': group_model.conf_int().loc[exposure_name].tolist(),
                'r_squared': group_model.rsquared,
                'n': len(group_data)
            }
    
    # Test for effect modification (interaction) using a combined model
    # Create interaction terms
    data_copy = data.copy()
    groups = data[modifier_name].unique()
    reference_group = groups[0]  # Use first group as reference
    
    # Create dummy variables for groups
    for group in groups[1:]:  # Skip reference group
        data_copy[f"{group}_dummy"] = (data_copy[modifier_name] == group).astype(int)
        # Create interaction terms
        data_copy[f"{exposure_name}_{group}"] = data_copy[exposure_name] * data_copy[f"{group}_dummy"]
    
    # Create model with interaction terms
    X_interaction = sm.add_constant(data_copy[[exposure_name] + 
                                             [f"{group}_dummy" for group in groups[1:]] +
                                             [f"{exposure_name}_{group}" for group in groups[1:]]])
    
    interaction_model = sm.OLS(data_copy[outcome_name], X_interaction).fit()
    
    # Extract p-values for interaction terms
    interaction_p_values = {}
    for group in groups[1:]:
        interaction_p_values[group] = interaction_model.pvalues[f"{exposure_name}_{group}"]
    
    results['interaction_p_values'] = interaction_p_values
    results['significant_interaction'] = any(p < 0.05 for p in interaction_p_values.values())
    
    return results

################
# Layout Setup #
################
st.set_page_config(layout="wide")

# Title and introduction
st.title("🔄 Effect Modification in Epidemiology")
st.markdown("""
This interactive module helps you understand how the effect of an exposure on an outcome
can vary across different subgroups, a phenomenon known as **effect modification** or 
**effect measure modification**.

**Effect modification** occurs when the effect of an exposure on an outcome varies across levels of 
a third variable. Unlike confounding, effect modification is not a bias but rather 
a real phenomenon that needs to be described and understood.
""")

#################################
# Controls & Data Visualization #
#################################

tab1, tab2, tab3, tab4 = st.tabs(["📊 Interactive Visualization", "📘 Educational Content", "🧠 Quiz & Exercises", "💻 Code Laboratory"])

with tab1:
    st.markdown("### Interactive Effect Modification Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Controls")
        
        # Scenario selector
        scenario = st.selectbox(
            "Choose a scenario:",
            options=["age_medication", "sex_exercise", "genotype_treatment"],
            format_func=lambda x: {
                "age_medication": "🩺 Medication effect by age group",
                "sex_exercise": "🏃 Exercise effect by sex",
                "genotype_treatment": "🧬 Treatment effect by genotype"
            }[x],
            help="Select a real-world scenario to explore"
        )
        
        # Overall effect modification strength
        effect_strength = st.slider(
            "Overall Effect Modification Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Controls the overall magnitude of effect differences between groups"
        )
        
        # Define group labels based on scenario
        if scenario == "age_medication":
            group_labels = ['Young', 'Middle', 'Old']
        elif scenario == "sex_exercise":
            group_labels = ['Male', 'Female']  # Biological sex categories
        elif scenario == "genotype_treatment":
            group_labels = ['AA', 'AB', 'BB']
        else:
            group_labels = ['Group 1', 'Group 2', 'Group 3']
        
        # Advanced controls in an expander
        with st.expander("Advanced Group-Specific Controls"):
            # Individual effect sizes by group
            group1_effect = st.slider(
                f"{group_labels[0]} Group Effect Size",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1
            )
            
            group2_effect = st.slider(
                f"{group_labels[1]} Group Effect Size",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            # Only show the third group slider if it exists for this scenario
            if len(group_labels) > 2:
                group3_effect = st.slider(
                    f"{group_labels[2]} Group Effect Size",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.1
                )
            else:
                # Default value if not used
                group3_effect = 0.0
        
        # Display options
        visualization_type = st.radio(
            "Visualization Type:",
            ["Scatter Plot", "Overlay Plot", "Effect Size Comparison"]
        )
        
        show_regression = st.checkbox("Show Regression Lines", value=True)
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
        
        # Generate data button
        if st.button("Generate New Data Sample"):
            st.session_state.regenerate_data = True
        else:
            if 'regenerate_data' not in st.session_state:
                st.session_state.regenerate_data = False
        
        st.markdown("""
        ### What to Look For:
        
        1. Different slopes across groups indicate effect modification
        2. The larger the difference in slopes, the stronger the effect modification
        3. When effect strength is 0, all groups have similar slopes
        4. Statistical significance is shown in the metrics panel
        """)
            
    with col2:
        # Generate data
        data = generate_effect_modification_data(
            n_samples=1000,
            effect_modifier_strength=effect_strength,
            group1_effect=group1_effect,
            group2_effect=group2_effect,
            group3_effect=group3_effect,
            scenario=scenario
        )
        
        # Extract column names
        cols = list(data.columns)
        modifier_name, exposure_name, outcome_name = cols[0], cols[1], cols[2]
        
        # Calculate statistics
        stats = calculate_group_statistics(data)
        
        # Display statistics in columns
        st.subheader("Effect Modification Metrics")
        metrics_cols = st.columns(len(data[modifier_name].unique()) + 1)
        
        # Overall column
        with metrics_cols[0]:
            st.metric("Overall Effect", f"{stats['overall']['coefficient']:.3f}")
            st.markdown(f"**CI**: [{stats['overall']['confidence_interval'][0]:.3f}, {stats['overall']['confidence_interval'][1]:.3f}]")
            st.markdown(f"**p-value**: {stats['overall']['p_value']:.3f}")
        
        # Group-specific columns
        for i, group in enumerate(sorted(data[modifier_name].unique())):
            with metrics_cols[i+1]:
                st.metric(
                    f"{group} Effect", 
                    f"{stats[group]['coefficient']:.3f}",
                    delta=f"{stats[group]['coefficient'] - stats['overall']['coefficient']:.3f}",
                    delta_color="off"
                )
                st.markdown(f"**CI**: [{stats[group]['confidence_interval'][0]:.3f}, {stats[group]['confidence_interval'][1]:.3f}]")
                st.markdown(f"**p-value**: {stats[group]['p_value']:.3f}")
        
        # Create appropriate visualization based on selection
        if visualization_type == "Faceted Plot":
            # Create faceted scatter plot
            fig = px.scatter(
                data,
                x=exposure_name,
                y=outcome_name,
                color=modifier_name,
                facet_col=modifier_name,
                title=f'Effect Modification by {modifier_name}',
                template="simple_white",
                trendline="ols" if show_regression else None,
                height=500
            )
            
            # Add custom annotations
            for i, group in enumerate(sorted(data[modifier_name].unique())):
                fig.add_annotation(
                    xref=f'x{i+1}', yref=f'y{i+1}',
                    x=0.5, y=0.9,
                    text=f"Effect: {stats[group]['coefficient']:.3f}",
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
        elif visualization_type == "Overlay Plot":
            # Create overlaid scatter plot
            fig = px.scatter(
                data,
                x=exposure_name,
                y=outcome_name,
                color=modifier_name,
                title=f'Effect Modification by {modifier_name} (Overlay)',
                template="simple_white",
                height=500
            )
            
            # Add regression lines
            if show_regression:
                for group in sorted(data[modifier_name].unique()):
                    group_data = data[data[modifier_name] == group]
                    
                    # Calculate regression
                    X = sm.add_constant(group_data[exposure_name])
                    model = sm.OLS(group_data[outcome_name], X).fit()
                    
                    # Create x range for prediction
                    x_range = np.linspace(data[exposure_name].min(), data[exposure_name].max(), 100)
                    y_pred = model.params[0] + model.params[1] * x_range
                    
                    # Add regression line
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_pred,
                            mode='lines',
                            name=f'{group} trend',
                            line=dict(width=3)
                        )
                    )
                    
                    # Add confidence intervals
                    if show_confidence:
                        # Prepare prediction data
                        X_pred = sm.add_constant(pd.Series(x_range))
                        
                        # Calculate prediction intervals
                        y_pred_ci = model.get_prediction(X_pred).conf_int(alpha=0.05)
                        
                        # Add confidence interval
                        fig.add_trace(
                            go.Scatter(
                                x=np.concatenate([x_range, x_range[::-1]]),
                                y=np.concatenate([y_pred_ci[:, 0], y_pred_ci[::-1, 1]]),
                                fill='toself',
                                fillcolor=f'rgba(128, 128, 128, 0.2)',
                                line=dict(color='rgba(255, 255, 255, 0)'),
                                hoverinfo='skip',
                                showlegend=False
                            )
                        )
            
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
        else:  # Effect Size Comparison
            # Create a bar chart to compare effect sizes
            groups = sorted(data[modifier_name].unique())
            coefficients = [stats[group]['coefficient'] for group in groups]
            ci_lower = [stats[group]['confidence_interval'][0] for group in groups]
            ci_upper = [stats[group]['confidence_interval'][1] for group in groups]
            
            # Calculate error margins for error bars
            error_y = np.array([
                np.array(coefficients) - np.array(ci_lower),
                np.array(ci_upper) - np.array(coefficients)
            ])
            
            # Create comparison figure
            fig = go.Figure()
            
            # Add bars for each group
            fig.add_trace(
                go.Bar(
                    x=groups,
                    y=coefficients,
                    text=[f"{coef:.3f}" for coef in coefficients],
                    textposition='auto',
                    marker_color=['#3498db', '#2ecc71', '#e74c3c'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error_y[1],
                        arrayminus=error_y[0],
                        visible=show_confidence
                    )
                )
            )
            
            # Add a line for the overall effect
            fig.add_shape(
                type="line",
                x0=-0.5, y0=stats['overall']['coefficient'],
                x1=len(groups)-0.5, y1=stats['overall']['coefficient'],
                line=dict(color="black", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=len(groups)-0.5,
                y=stats['overall']['coefficient'],
                text=f"Overall: {stats['overall']['coefficient']:.3f}",
                showarrow=False,
                xanchor="right",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            # Update layout
            fig.update_layout(
                title='Effect Size Comparison Across Groups',
                xaxis_title=modifier_name,
                yaxis_title=f'Effect Size (Regression Coefficient)',
                template="simple_white",
                height=500
            )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical significance of effect modification
        significance_text = "⚠️ **Statistically significant** effect modification detected (p < 0.05)" if stats['significant_interaction'] else "No statistically significant effect modification detected"
        
        significance_color = "success" if stats['significant_interaction'] else "warning"
        st.markdown(f":{significance_color}[{significance_text}]")
        
        # Group interaction p-values
        st.markdown("#### Statistical Tests for Effect Modification")
        
        p_value_cols = st.columns(len(stats['interaction_p_values']))
        for i, (group, p_value) in enumerate(stats['interaction_p_values'].items()):
            with p_value_cols[i]:
                st.metric(
                    f"{group} vs Reference", 
                    f"p = {p_value:.3f}",
                    delta="Significant" if p_value < 0.05 else "Not Significant",
                    delta_color="normal" if p_value < 0.05 else "off"
                )

with tab2:
    st.markdown("## Understanding Effect Modification")
    
    st.markdown("""
    ### What is Effect Modification?
    
    **Effect modification** (also called effect measure modification or EMM) occurs when the effect 
    of an exposure on an outcome **varies across levels of a third variable**. This means that the 
    magnitude, and sometimes even the direction, of the association between the exposure and outcome 
    differs depending on the value of this third variable.
    
    ### Key Characteristics:
    
    1. **Not a bias**: Unlike confounding, effect modification is not a bias that needs to be 
       eliminated. It is a real phenomenon that should be described and understood.
    
    2. **Measure-specific**: Effect modification can be present on one scale (e.g., relative risk) 
       but absent on another scale (e.g., risk difference).
    
    3. **Statistical interaction**: In regression models, effect modification is often modeled using 
       interaction terms.
    
    4. **Important for targeted interventions**: Understanding effect modification helps target 
       interventions to groups who will benefit most.
    """)
    
    st.markdown("### Real-world Examples")
    
    # Create tabs for different examples
    ex1, ex2, ex3 = st.tabs(["Example 1: Age & Medication", "Example 2: Sex & Exercise", "Example 3: Genotype & Treatment"])
    
    with ex1:
        st.markdown("""
        #### Age as an Effect Modifier for Medication Response
        
        **Scenario**: The effect of a blood pressure medication varies by age group.
        
        **Findings**:
        - **Young adults (18-40)**: Small reduction in blood pressure (effect size 0.2)
        - **Middle-aged (41-65)**: Moderate reduction in blood pressure (effect size 0.5)
        - **Elderly (65+)**: Large reduction in blood pressure (effect size 0.8)
        
        **Implications**:
        - Medication dosing may need to be adjusted by age group
        - Risk-benefit calculations should consider age-specific effects
        - Clinical trials should include age-stratified analyses
        """)
        
        st.image("https://www.statnews.com/wp-content/uploads/2019/06/hypertension-and-heart-disease-graphic-1024x576.jpg", width=500, caption="Blood pressure varies with age and affects medication efficacy")
    
    with ex2:
        st.markdown("""
        #### Sex as an Effect Modifier for Exercise Impact
        
        **Scenario**: The effect of exercise intensity on weight loss differs between biological males and females.
        
        **Findings**:
        - **Males**: Strong relationship between exercise intensity and weight loss (effect size 0.8)
        - **Females**: Moderate relationship between exercise intensity and weight loss (effect size 0.5)
        
        **Possible explanations**:
        - Physiological differences in metabolism and metabolic rate
        - Differences in muscle mass and body composition (lean mass vs. fat mass)
        - Hormonal differences affecting fat storage and utilization
        - Differences in substrate utilization during exercise
        
        **Implications**:
        - Exercise programs may need sex-specific modifications
        - Weight loss expectations should consider biological sex differences
        - Research studies should analyze results separately by sex
        - Personalized exercise prescriptions should account for biological sex
        """)
    
    with ex3:
        st.markdown("""
        #### Genotype as an Effect Modifier for Drug Response
        
        **Scenario**: The efficacy of a cancer drug depends on a specific genetic marker.
        
        **Findings**:
        - **Genotype AA**: Strong tumor reduction (effect size 0.8)
        - **Genotype AB**: Moderate tumor reduction (effect size 0.5)
        - **Genotype BB**: Minimal tumor reduction (effect size 0.2)
        
        **Implications**:
        - Precision medicine approach: genetic testing before prescription
        - Drug development may target specific genetic profiles
        - Clinical trials should include genetic stratification
        
        This is the foundation of **pharmacogenomics** - how genetic variation affects individual responses to medications.
        """)
    
    st.markdown("""
    ### Effect Modification vs. Confounding vs. Interaction
    
    | Concept | Definition | How to Handle |
    |---------|------------|---------------|
    | **Effect Modification** | The effect of exposure varies across levels of a third variable | Describe and report stratum-specific effects |
    | **Confounding** | A third variable distorts the observed association between exposure and outcome | Control for it through study design or analysis |
    | **Statistical Interaction** | A statistical modeling term where the joint effect of two exposures differs from their individual effects | Include interaction terms in regression models |
    
    ### Methodological Considerations:
    
    1. **A priori identification**: Potential effect modifiers should ideally be specified before analysis.
    
    2. **Adequate sample size**: Testing for effect modification requires larger sample sizes than main effect analyses.
    
    3. **Multiple testing**: Testing numerous potential effect modifiers increases the risk of false positives.
    
    4. **Scale dependence**: Effect modification can appear or disappear depending on the scale of measurement.
    """)

with tab3:
    st.markdown("## 🧠 Test Your Understanding")
    
    # Multiple quiz questions with increasing complexity
    st.subheader("Quiz Questions")
    
    q1 = st.radio(
        "1. Which statement best describes effect modification?",
        [
            "A bias that distorts the exposure-outcome relationship",
            "A phenomenon where the effect of exposure is stronger/weaker in different subgroups",
            "A statistical error that should be controlled for",
            "A type of measurement error in epidemiological studies"
        ],
        key="q1"
    )

    if q1 == "A phenomenon where the effect of exposure is stronger/weaker in different subgroups":
        st.success("✅ Correct! Effect modification means the exposure-outcome effect differs across strata of a third variable.")
    elif q1:  # Only show feedback if an answer has been selected
        st.error("❌ That's not right. Effect modification is a real phenomenon, not a bias or error.")
    
    q2 = st.radio(
        "2. In the simulation, what happens when you set all group-specific effects to be equal?",
        [
            "Effect modification disappears and all groups show similar slopes",
            "Effect modification remains but becomes statistically insignificant",
            "The average effect across all groups increases",
            "The statistical power to detect effects decreases"
        ],
        key="q2"
    )

    if q2 == "Effect modification disappears and all groups show similar slopes":
        st.success("✅ Correct! When all groups have the same effect size, there is no effect modification.")
    elif q2:
        st.error("❌ Try again. Think about what equal effects across groups means conceptually.")
    
    q3 = st.radio(
        "3. Why is effect modification important in public health?",
        [
            "It helps identify and remove biases in study designs",
            "It allows for more targeted interventions in subgroups that benefit most",
            "It increases the statistical power of epidemiological studies",
            "It eliminates the need for adjusting for confounding variables"
        ],
        key="q3"
    )

    if q3 == "It allows for more targeted interventions in subgroups that benefit most":
        st.success("✅ Correct! Understanding effect modification helps target interventions to those who will benefit most.")
    elif q3:
        st.error("❌ Not quite. Think about how knowing different effects in different groups might inform intervention strategies.")
    
    q4 = st.radio(
        "4. How is effect modification typically modeled in regression analysis?",
        [
            "By removing the effect modifier from the dataset",
            "By including interaction terms between the exposure and the effect modifier",
            "By conducting separate analyses and not including the effect modifier",
            "By controlling for the effect modifier as a confounder"
        ],
        key="q4"
    )

    if q4 == "By including interaction terms between the exposure and the effect modifier":
        st.success("✅ Correct! Interaction terms in regression models allow us to quantify how the effect of the exposure varies across levels of the effect modifier.")
    elif q4:
        st.error("❌ That's not the standard approach. Think about how statistical models can represent varying effects.")
    
    # Interactive exercise
    st.subheader("Interactive Exercise")
    
    st.markdown("""
    ### Predict the Pattern
    
    Imagine a study of a new pain medication where age is an effect modifier. The medication works better in younger people.
    
    If you were to create a scatter plot with "Medication Dose" on the x-axis and "Pain Reduction" on the y-axis for different age groups, 
    which pattern would you expect to see?
    """)
    
    answer = st.radio(
        "Select the most likely pattern:",
        [
            "All age groups show identical slopes for dose-response relationship",
            "Younger groups show steeper slopes than older groups",
            "Older groups show steeper slopes than younger groups",
            "The slopes go in opposite directions for different age groups"
        ],
        key="exercise"
    )
    
    if answer == "Younger groups show steeper slopes than older groups":
        st.success("""
        ✅ Correct! If the medication works better in younger people, we would expect to see steeper slopes in the younger groups, 
        indicating a stronger relationship between dose and pain reduction for them.
        
        This is a classic pattern of effect modification - the same exposure (medication) has different effects (pain reduction) 
        depending on the third variable (age group).
        """)
    elif answer:
        st.error("❌ That's not what we would expect based on the scenario. Try again!")
    
    # Real-world application
    st.subheader("Real-world Application Exercise")
    
    st.markdown("""
    ### Case Study Analysis
    
    **Study Findings**: A weight loss program shows the following results:
    
    - In people with normal metabolism: 1 hour of exercise correlates with 0.5 kg weight loss per week
    - In people with slow metabolism: 1 hour of exercise correlates with 0.2 kg weight loss per week
    
    **Question**: As a health advisor, how would you use this information about effect modification?
    """)
    
    options = st.multiselect(
        "Select all appropriate actions:",
        [
            "Recommend the same exercise program to everyone regardless of metabolism",
            "Advise those with slow metabolism that they may need more exercise for similar results",
            "Recommend additional dietary modifications for those with slow metabolism",
            "Conclude that exercise is not effective for weight loss in people with slow metabolism",
            "Design different exercise targets based on metabolism type",
            "Ignore these findings as they likely represent confounding rather than effect modification"
        ],
        key="case_study"
    )
    
    correct_options = [
        "Advise those with slow metabolism that they may need more exercise for similar results",
        "Recommend additional dietary modifications for those with slow metabolism",
        "Design different exercise targets based on metabolism type"
    ]
    
    if set(options) == set(correct_options):
        st.success("✅ Perfect! You've correctly identified all the appropriate actions.")
    elif any(option in correct_options for option in options) and all(option in correct_options for option in options):
        st.warning("⚠️ You're on the right track, but you haven't selected all the appropriate actions.")
    elif options:
        st.error("❌ Some of your selected actions aren't appropriate. Remember that effect modification means different approaches may be needed for different groups.")

with tab4:
    effect_modification_code.app()

# Footer with references
st.markdown("""
---
### References

1. VanderWeele TJ, Knol MJ. (2014). A Tutorial on Interaction. *Epidemiologic Methods*, 3(1), 33-72.
2. Rothman KJ, Greenland S, Lash TL. (2008). *Modern Epidemiology*. 3rd Edition. Philadelphia: Lippincott Williams & Wilkins.
3. Szklo M, Nieto FJ. (2014). *Epidemiology: Beyond the Basics*. 3rd Edition. Jones & Bartlett Learning.
4. Greenland S. (1993). Basic Problems in Interaction Assessment. *Environmental Health Perspectives*, 101(Suppl 4), 59-66.

---
*Created for educational purposes. This interactive tool helps visualize effect modification concepts in epidemiology.*
""")