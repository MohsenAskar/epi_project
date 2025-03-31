import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import io
from contextlib import redirect_stdout
import sys
import traceback

def app():
    st.title("Interactive Coding Laboratory: Effect Modification Analysis")
    
    st.markdown("""
    ## Learn by Coding: Effect Modification Concepts
    
    This interactive coding laboratory allows you to modify and execute Python code directly in your browser.
    Experiment with different aspects of effect modification by modifying the example code and seeing the results.
    
    Choose a topic to explore:
    """)
    
    # Topic selection
    topic = st.selectbox(
        "Select an effect modification topic:",
        ["Basic Effect Modification Analysis", 
         "Interaction Terms in Regression",
         "Stratified Analysis Techniques",
         "Real-world Public Health Examples"]
    )
    
    # Display the selected topic
    if topic == "Basic Effect Modification Analysis":
        basic_effect_modification_lesson()
    elif topic == "Interaction Terms in Regression":
        interaction_terms_lesson()
    elif topic == "Stratified Analysis Techniques":
        stratified_analysis_lesson()
    elif topic == "Real-world Public Health Examples":
        public_health_examples_lesson()

def execute_code(code_string):
    """
    Safely execute the provided code string and capture its output
    """
    # Create string buffer to capture print statements
    buffer = io.StringIO()
    
    # Dictionary to store variables that will be returned for plotting or further use
    output_vars = {}
    
    try:
        # Execute the code with stdout redirected to our buffer
        with redirect_stdout(buffer):
            # Create a local environment with necessary imports
            exec_globals = {
                'np': np,
                'pd': pd,
                'plt': go,
                'px': px,
                'go': go,
                'sm': sm,
                'make_subplots': make_subplots,
                'output_vars': output_vars
            }
            
            # Execute the code
            exec(code_string, exec_globals)
            
            # Save any variables the user assigned to output_vars dictionary
            output_vars = exec_globals['output_vars']
        
        # Get the printed output
        output = buffer.getvalue()
        
        return True, output, output_vars
    
    except Exception as e:
        # Return the error message
        error_msg = traceback.format_exc()
        return False, error_msg, {}
    
    finally:
        buffer.close()

def basic_effect_modification_lesson():
    st.header("Basic Effect Modification Analysis")
    
    st.markdown("""
    ### Understanding Effect Modification
    
    Effect modification occurs when the effect of an exposure on an outcome varies across levels of a third variable.
    In this example, we'll simulate data where the effect of a medication on treatment outcome varies by age group.
    
    Let's explore how to generate and analyze such data:
    """)
    
    # Initial code example
    initial_code = """# Basic effect modification analysis
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 300       # Number of subjects
age_effect_young = 0.2   # Effect size for young group
age_effect_middle = 0.5  # Effect size for middle-aged group
age_effect_old = 0.8     # Effect size for elderly group
noise_level = 0.5     # Level of random variation

# Set random seed for reproducibility
np.random.seed(42)

# Generate the effect modifier (age group)
age_groups = ['Young', 'Middle', 'Old']
age = np.random.choice(age_groups, n_samples)

# Generate the exposure (medication dose)
medication_dose = np.random.normal(0, 1, n_samples)

# Generate the outcome (treatment effect) with different effects by age group
treatment_effect = np.zeros(n_samples)

# Apply different effects based on age group
for i, group in enumerate(age_groups):
    mask = (age == group)
    
    # Get the right effect size for this age group
    if group == 'Young':
        effect = age_effect_young
    elif group == 'Middle':
        effect = age_effect_middle
    else:  # Old
        effect = age_effect_old
    
    # Generate outcome with age-specific effect
    treatment_effect[mask] = (
        medication_dose[mask] * effect + 
        np.random.normal(0, noise_level, sum(mask))
    )

# Create a DataFrame
data = pd.DataFrame({
    'Age Group': age,
    'Medication Dose': medication_dose,
    'Treatment Effect': treatment_effect
})

# Print summary statistics by group
print("Sample size by group:")
print(data.groupby('Age Group').size())
print("\\nMean treatment effect by group:")
print(data.groupby('Age Group')['Treatment Effect'].mean().round(3))

# Calculate effect size (regression coefficient) by group
group_effects = {}

for group in age_groups:
    group_data = data[data['Age Group'] == group]
    X = sm.add_constant(group_data['Medication Dose'])
    model = sm.OLS(group_data['Treatment Effect'], X).fit()
    
    group_effects[group] = {
        'coefficient': model.params[1],
        'std_error': model.bse[1],
        'p_value': model.pvalues[1],
        'conf_int': model.conf_int().iloc[1].tolist()
    }
    
    print(f"\\nRegression results for {group} group:")
    print(f"Effect size: {model.params[1]:.3f}")
    print(f"p-value: {model.pvalues[1]:.4f}")
    print(f"95% CI: [{model.conf_int().iloc[1, 0]:.3f}, {model.conf_int().iloc[1, 1]:.3f}]")

# Test for effect modification using an interaction model
# Create dummy variables for age groups
data['is_middle'] = (data['Age Group'] == 'Middle').astype(int)
data['is_old'] = (data['Age Group'] == 'Old').astype(int)

# Create interaction terms
data['dose_middle'] = data['Medication Dose'] * data['is_middle']
data['dose_old'] = data['Medication Dose'] * data['is_old']

# Fit model with interaction terms
X_interaction = sm.add_constant(data[['Medication Dose', 'is_middle', 'is_old', 
                                      'dose_middle', 'dose_old']])
interaction_model = sm.OLS(data['Treatment Effect'], X_interaction).fit()

print("\\nInteraction model results:")
print(interaction_model.summary().tables[1])

# Check if effect modification is statistically significant
em_significant = (interaction_model.pvalues['dose_middle'] < 0.05 or 
                  interaction_model.pvalues['dose_old'] < 0.05)

if em_significant:
    print("\\n✓ Effect modification is statistically significant (p < 0.05)")
else:
    print("\\n✗ Effect modification is not statistically significant (p >= 0.05)")

# Create a scatter plot with regression lines
fig = px.scatter(
    data, 
    x='Medication Dose', 
    y='Treatment Effect', 
    color='Age Group',
    title='Effect Modification: How Medication Effect Varies by Age Group',
    color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c']
)

# Add regression lines
for group in age_groups:
    group_data = data[data['Age Group'] == group]
    
    # Get regression results
    coef = group_effects[group]['coefficient']
    intercept = sm.OLS(group_data['Treatment Effect'], 
                      sm.add_constant(group_data['Medication Dose'])).fit().params[0]
    
    # Create x range for prediction
    x_range = np.linspace(data['Medication Dose'].min(), data['Medication Dose'].max(), 100)
    y_pred = intercept + coef * x_range
    
    # Add the line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'{group} (effect={coef:.2f})',
            line=dict(width=3)
        )
    )

# Update layout
fig.update_layout(
    xaxis_title='Medication Dose',
    yaxis_title='Treatment Effect',
    legend_title='Age Group',
    height=500
)

# Store the figure in output_vars to display it
output_vars['fig'] = fig

# Create a bar chart comparing effect sizes
fig2 = go.Figure()

# Add bars for each group
coefficients = [group_effects[group]['coefficient'] for group in age_groups]
errors = [group_effects[group]['std_error'] for group in age_groups]

fig2.add_trace(
    go.Bar(
        x=age_groups,
        y=coefficients,
        error_y=dict(
            type='data',
            array=errors,
            visible=True
        ),
        marker_color=['#3498db', '#2ecc71', '#e74c3c']
    )
)

# Add a reference line at y=0
fig2.add_shape(
    type="line",
    x0=-0.5, y0=0, x1=2.5, y1=0,
    line=dict(color="black", width=1, dash="dash")
)

# Update layout
fig2.update_layout(
    title='Comparison of Effect Sizes by Age Group',
    xaxis_title='Age Group',
    yaxis_title='Effect Size (Regression Coefficient)',
    height=400
)

# Store the second figure
output_vars['fig2'] = fig2
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and click Execute to see the results:", 
                           value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Change the effect sizes (`age_effect_young`, `age_effect_middle`, `age_effect_old`) to see how the group differences change
            - Set all effect sizes to the same value to see what happens when there's no effect modification
            - Increase or decrease the `noise_level` to see how it affects the detection of effect modification
            - Try a negative effect in one group and a positive effect in another
            
            **Challenges:**
            1. Create a scenario where effect modification is not statistically significant
            2. Make the effect size in the 'Young' group zero (no effect)
            3. Add a fourth age group with its own effect size
            4. Increase sample size to see how it affects statistical power to detect effect modification
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Output:")
                st.text(output)
            
            # Display any figures generated
            if 'fig' in output_vars:
                st.plotly_chart(output_vars['fig'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Concepts")
    st.markdown("""
    ### Understanding Effect Modification in the Code:
    
    1. **Data Generation**:
       - We create a scenario where medication has different effects in different age groups
       - The `effect` parameter controls how strongly medication dose affects the outcome
       - Distinct effect sizes by group are the essence of effect modification
    
    2. **Detection Methods**:
       - Group-specific regression: Analyzing each subgroup separately
       - Interaction terms: Testing if the interaction coefficients are significant
       - Visualization: Different slopes indicate effect modification
    
    3. **Statistical Testing**:
       - The p-values for interaction terms tell us if effect modification is statistically significant
       - Small p-values (<0.05) for interaction terms suggest real effect modification
       - Sample size affects our power to detect these differences
    
    4. **Interpretation**:
       - Effect modification means we can't summarize the exposure effect with a single number
       - We need to report group-specific effects
       - This informs personalized treatment approaches (e.g., medication dosing by age)
    """)

def interaction_terms_lesson():
    st.header("Interaction Terms in Regression")
    
    st.markdown("""
    ### Modeling Effect Modification with Interaction Terms
    
    In regression analysis, effect modification is typically modeled using **interaction terms**.
    An interaction term is created by multiplying two variables together, allowing the effect
    of one variable to depend on the value of another.
    
    Let's explore how to implement and interpret interaction terms:
    """)
    
    # Initial code example
    initial_code = """# Modeling effect modification with interaction terms
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 500         # Number of observations
main_effect = 0.5       # Main effect of exposure
modifier_effect = 0.3   # Main effect of modifier 
interaction_strength = 0.7  # Strength of interaction effect
binary_modifier = True  # Whether modifier is binary (True) or continuous (False)
noise_level = 1.0       # Level of random variation

# Set random seed for reproducibility
np.random.seed(42)

# Generate data
exposure = np.random.normal(0, 1, n_samples)

# Generate modifier variable (either binary or continuous)
if binary_modifier:
    modifier = np.random.binomial(1, 0.5, n_samples)
    modifier_name = 'Binary Modifier'
else:
    modifier = np.random.normal(0, 1, n_samples)
    modifier_name = 'Continuous Modifier'

# Generate outcome with interaction effect
outcome = (
    main_effect * exposure +  # Main effect of exposure
    modifier_effect * modifier +  # Main effect of modifier
    interaction_strength * exposure * modifier +  # Interaction effect
    np.random.normal(0, noise_level, n_samples)  # Random noise
)

# Create DataFrame
data = pd.DataFrame({
    'Exposure': exposure,
    modifier_name: modifier,
    'Outcome': outcome
})

# Create interaction term
data['Interaction'] = data['Exposure'] * data[modifier_name]

# Fit regression model with interaction
X = sm.add_constant(data[['Exposure', modifier_name, 'Interaction']])
model = sm.OLS(data['Outcome'], X).fit()

# Print model results
print("Regression Model with Interaction Term:")
print(model.summary().tables[1])

# Check if interaction is significant
is_significant = model.pvalues['Interaction'] < 0.05
significance_note = "statistically significant" if is_significant else "not statistically significant"
print(f"\\nThe interaction term is {significance_note} (p = {model.pvalues['Interaction']:.4f})")

# Interpretation of coefficients
print("\\nInterpretation of Coefficients:")
print(f"Main effect of exposure: {model.params['Exposure']:.3f}")
print(f"  - This represents the effect of exposure when {modifier_name} = 0")

if binary_modifier:
    print(f"\\nEffect of exposure when {modifier_name} = 0: {model.params['Exposure']:.3f}")
    print(f"Effect of exposure when {modifier_name} = 1: {model.params['Exposure'] + model.params['Interaction']:.3f}")
    print(f"Difference in effects (interaction): {model.params['Interaction']:.3f}")
else:
    print("\\nEffect of exposure at different levels of the modifier:")
    for mod_value in [-1, 0, 1]:
        effect = model.params['Exposure'] + model.params['Interaction'] * mod_value
        print(f"  When {modifier_name} = {mod_value}: {effect:.3f}")

# Create visualization
if binary_modifier:
    # For binary modifier: Create faceted scatter plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f'{modifier_name} = 0', f'{modifier_name} = 1'])
    
    # Group 0
    group0 = data[data[modifier_name] == 0]
    fig.add_trace(
        go.Scatter(
            x=group0['Exposure'],
            y=group0['Outcome'],
            mode='markers',
            marker=dict(color='blue'),
            name=f'{modifier_name} = 0'
        ),
        row=1, col=1
    )
    
    # Add regression line for group 0
    x_range = np.linspace(data['Exposure'].min(), data['Exposure'].max(), 100)
    y_pred = model.params['const'] + model.params['Exposure'] * x_range
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(color='blue', width=3),
            name=f'Effect = {model.params["Exposure"]:.2f}'
        ),
        row=1, col=1
    )
    
    # Group 1
    group1 = data[data[modifier_name] == 1]
    fig.add_trace(
        go.Scatter(
            x=group1['Exposure'],
            y=group1['Outcome'],
            mode='markers',
            marker=dict(color='red'),
            name=f'{modifier_name} = 1'
        ),
        row=1, col=2
    )
    
    # Add regression line for group 1
    y_pred = (model.params['const'] + model.params[modifier_name] + 
              (model.params['Exposure'] + model.params['Interaction']) * x_range)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Effect = {model.params["Exposure"] + model.params["Interaction"]:.2f}'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Effect Modification with Binary Modifier',
        height=500,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='Exposure', row=1, col=1)
    fig.update_xaxes(title_text='Exposure', row=1, col=2)
    fig.update_yaxes(title_text='Outcome', row=1, col=1)
    
else:
    # For continuous modifier: Create 3D plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=data['Exposure'],
            y=data[modifier_name],
            z=data['Outcome'],
            mode='markers',
            marker=dict(
                size=5,
                color=data['Outcome'],
                colorscale='Viridis',
                opacity=0.8
            )
        )
    ])
    
    # Create a mesh grid for the prediction surface
    x_range = np.linspace(data['Exposure'].min(), data['Exposure'].max(), 20)
    y_range = np.linspace(data[modifier_name].min(), data[modifier_name].max(), 20)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    
    # Calculate predicted values
    z_mesh = (model.params['const'] + 
              model.params['Exposure'] * x_mesh + 
              model.params[modifier_name] * y_mesh + 
              model.params['Interaction'] * x_mesh * y_mesh)
    
    # Add prediction surface
    fig.add_trace(
        go.Surface(
            x=x_range,
            y=y_range,
            z=z_mesh,
            opacity=0.7,
            colorscale='Viridis',
            name='Prediction Surface'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Effect Modification with Continuous Modifier',
        scene=dict(
            xaxis_title='Exposure',
            yaxis_title=modifier_name,
            zaxis_title='Outcome'
        ),
        height=700
    )

# Store the figure in output_vars to display it
output_vars['fig'] = fig

# Create a second visualization showing the changing slopes
if not binary_modifier:
    # Create a plot showing how the effect changes with modifier value
    modifier_values = np.linspace(data[modifier_name].min(), data[modifier_name].max(), 100)
    effect_sizes = model.params['Exposure'] + model.params['Interaction'] * modifier_values
    
    fig2 = go.Figure()
    
    # Add the line showing changing effect size
    fig2.add_trace(
        go.Scatter(
            x=modifier_values,
            y=effect_sizes,
            mode='lines',
            line=dict(color='blue', width=3)
        )
    )
    
    # Add a reference line at y=0
    fig2.add_shape(
        type="line",
        x0=data[modifier_name].min(), y0=0, 
        x1=data[modifier_name].max(), y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Update layout
    fig2.update_layout(
        title=f'How the Effect of Exposure Changes with {modifier_name}',
        xaxis_title=modifier_name,
        yaxis_title='Effect of Exposure on Outcome',
        height=400
    )
    
    # Store the second figure
    output_vars['fig2'] = fig2
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and click Execute to see the results:", 
                           value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Toggle `binary_modifier` between True and False to see different types of effect modification
            - Change `interaction_strength` to see how it affects the differences between groups
            - Set `interaction_strength = 0` to see what happens when there's no effect modification
            - Adjust `noise_level` to see how noise affects detection of interaction effects
            
            **Challenges:**
            1. Create a scenario where the effect reverses direction based on the modifier value
            2. Find the minimum `interaction_strength` needed for a statistically significant result
            3. Make the `main_effect` zero and see if the interaction is still detectable
            4. Create a scenario where the effect of exposure is significant only in one group
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Output:")
                st.text(output)
            
            # Display any figures generated
            if 'fig' in output_vars:
                st.plotly_chart(output_vars['fig'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Concepts")
    st.markdown("""
    ### Interaction Terms in Regression Models:
    
    1. **Mathematical Representation**:
       - Basic regression: Y = β₀ + β₁X + β₂Z + ε
       - With interaction: Y = β₀ + β₁X + β₂Z + β₃(X×Z) + ε
       - The coefficient β₃ represents how much the effect of X changes for each unit increase in Z
    
    2. **Interpretation of Coefficients**:
       - Main effect (β₁): Effect of X when Z = 0
       - Interaction term (β₃): How the effect of X changes as Z changes
       - Total effect of X: β₁ + β₃Z (depends on the value of Z)
    
    3. **Testing for Effect Modification**:
       - Null hypothesis: β₃ = 0 (no effect modification)
       - The p-value for the interaction term tells us whether effect modification is statistically significant
    
    4. **Visualizing Interactions**:
       - For binary modifiers: Compare slopes between groups
       - For continuous modifiers: 3D surface or plot of changing effect sizes
    
    5. **Power Considerations**:
       - Detecting interactions typically requires larger sample sizes than detecting main effects
       - Noisy data makes it harder to detect significant interactions
    """)

def stratified_analysis_lesson():
    st.header("Stratified Analysis Techniques")
    
    st.markdown("""
    ### Exploring Effect Modification through Stratification
    
    Stratified analysis is a fundamental approach to investigating effect modification.
    By examining the relationship between exposure and outcome separately within each level
    of the potential effect modifier, we can identify differences in effect sizes.
    
    Let's explore stratified analysis techniques:
    """)
    
    # Initial code example
    initial_code = """# Stratified analysis for effect modification
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 1000        # Number of observations
stratify_by = "ses"     # Options: "ses", "age", "sex", "region"
n_strata = 3            # Number of strata (levels of the stratification variable)
stratum_effects = [0.2, 0.5, 0.8]  # Effect in each stratum
include_confounding = False  # Whether to include a confounder
sample_imbalance = 0.0  # 0.0 = balanced samples, higher values = more imbalance

# Set random seed for reproducibility
np.random.seed(42)

# Configure variables based on stratification choice
if stratify_by == "ses":
    strata_names = ["Low SES", "Medium SES", "High SES"]
    exposure_name = "Education (years)"
    outcome_name = "Income (thousands)"
elif stratify_by == "age":
    strata_names = ["Young", "Middle-aged", "Elderly"]
    exposure_name = "Treatment Dose"
    outcome_name = "Symptom Score"
elif stratify_by == "sex":
    strata_names = ["Male", "Female"]
    exposure_name = "Exercise (hours/week)"
    outcome_name = "Weight Loss (kg)"
    n_strata = 2  # Override to 2 for sex
    stratum_effects = stratum_effects[:2]  # Use only first two effects
elif stratify_by == "region":
    strata_names = ["Region A", "Region B", "Region C"]
    exposure_name = "Policy Strength"
    outcome_name = "Health Metric"
else:
    strata_names = [f"Stratum {i+1}" for i in range(n_strata)]
    exposure_name = "Exposure"
    outcome_name = "Outcome"

# Ensure we have the right number of stratum names and effects
strata_names = strata_names[:n_strata]
stratum_effects = stratum_effects[:n_strata]

# Generate stratification variable with potentially imbalanced strata
if sample_imbalance > 0:
    # Create imbalanced probabilities
    probs = np.ones(n_strata)
    for i in range(n_strata):
        probs[i] = 1 + (i * sample_imbalance)
    probs = probs / probs.sum()  # Normalize to sum to 1
    
    # Generate strata with imbalance
    strata_idx = np.random.choice(range(n_strata), n_samples, p=probs)
else:
    # Balanced strata
    strata_idx = np.random.choice(range(n_strata), n_samples)

strata = np.array(strata_names)[strata_idx]

# Generate exposure
exposure = np.random.normal(0, 1, n_samples)

# Generate confounder if included
if include_confounding:
    confounder = 0.5 * np.random.normal(0, 1, n_samples)
    
    # Make exposure depend on confounder
    exposure = exposure + 0.7 * confounder
    
    confounder_name = "Confounder"
else:
    confounder = None
    confounder_name = None

# Generate outcome with different effects by stratum
outcome = np.zeros(n_samples)

# Apply different effects based on stratum
for i in range(n_strata):
    mask = (strata_idx == i)
    effect = stratum_effects[i]
    
    outcome_for_stratum = exposure[mask] * effect
    
    # Add confounder effect if included
    if include_confounding:
        outcome_for_stratum += 0.6 * confounder[mask]
    
    # Add random noise
    outcome_for_stratum += np.random.normal(0, 1, sum(mask))
    
    outcome[mask] = outcome_for_stratum

# Create DataFrame
data_dict = {
    'Stratum': strata,
    exposure_name: exposure,
    outcome_name: outcome
}

if include_confounding:
    data_dict[confounder_name] = confounder

data = pd.DataFrame(data_dict)

# Print sample size by stratum
print("Sample size by stratum:")
stratum_counts = data['Stratum'].value_counts().sort_index()
for stratum, count in zip(strata_names, stratum_counts):
    print(f"{stratum}: {count}")

# Perform stratified analysis
print("\\nStratified Analysis Results:")
print("-" * 50)

# Store results for visualization
stratum_results = {}

for stratum in strata_names:
    # Filter data for this stratum
    stratum_data = data[data['Stratum'] == stratum]
    
    # Simple regression model
    X = sm.add_constant(stratum_data[exposure_name])
    model_simple = sm.OLS(stratum_data[outcome_name], X).fit()
    
    # Store results
    stratum_results[stratum] = {
        'coefficient': model_simple.params[1],
        'std_err': model_simple.bse[1],
        'p_value': model_simple.pvalues[1],
        'conf_int': model_simple.conf_int().iloc[1].tolist(),
        'n': len(stratum_data)
    }
    
    # Print results
    print(f"Results for {stratum}:")
    print(f"  Sample size: {len(stratum_data)}")
    print(f"  Effect size: {model_simple.params[1]:.3f}")
    print(f"  95% CI: [{model_simple.conf_int().iloc[1, 0]:.3f}, {model_simple.conf_int().iloc[1, 1]:.3f}]")
    print(f"  p-value: {model_simple.pvalues[1]:.4f}")
    
    # If confounding is included, also show adjusted results
    if include_confounding:
        # Adjusted model with confounder
        X_adj = sm.add_constant(stratum_data[[exposure_name, confounder_name]])
        model_adj = sm.OLS(stratum_data[outcome_name], X_adj).fit()
        
        # Store adjusted results
        stratum_results[stratum]['adj_coefficient'] = model_adj.params[1]
        stratum_results[stratum]['adj_std_err'] = model_adj.bse[1]
        stratum_results[stratum]['adj_p_value'] = model_adj.pvalues[1]
        stratum_results[stratum]['adj_conf_int'] = model_adj.conf_int().iloc[1].tolist()
        
        print(f"  Adjusted effect size: {model_adj.params[1]:.3f}")
        print(f"  Adjusted 95% CI: [{model_adj.conf_int().iloc[1, 0]:.3f}, {model_adj.conf_int().iloc[1, 1]:.3f}]")
        print(f"  Adjusted p-value: {model_adj.pvalues[1]:.4f}")
    
    print("-" * 50)

# Test for heterogeneity of effects
# Fit model with interaction terms for all strata
# Create dummy variables for strata (first stratum is reference)
for i, stratum in enumerate(strata_names[1:], 1):
    data[f'is_{stratum}'] = (data['Stratum'] == stratum).astype(int)
    
    # Create interaction terms
    data[f'{exposure_name}_{stratum}'] = data[exposure_name] * data[f'is_{stratum}']

# Variables for the model
interaction_vars = [f'{exposure_name}_{stratum}' for stratum in strata_names[1:]]
stratum_vars = [f'is_{stratum}' for stratum in strata_names[1:]]
main_vars = [exposure_name]

if include_confounding:
    main_vars.append(confounder_name)

# Fit interaction model
X_interaction = sm.add_constant(data[main_vars + stratum_vars + interaction_vars])
interaction_model = sm.OLS(data[outcome_name], X_interaction).fit()

# Check if any interaction is significant
interaction_p_values = [interaction_model.pvalues[var] for var in interaction_vars]
significant_interaction = any(p < 0.05 for p in interaction_p_values)

print("Test for heterogeneity of effects (effect modification):")
print(f"Interactions p-values: {[f'{p:.4f}' for p in interaction_p_values]}")

if significant_interaction:
    print("✓ Evidence of significant effect modification (heterogeneity across strata)")
else:
    print("✗ No significant evidence of effect modification")

# Create a Forest Plot to visualize stratified results
fig = go.Figure()

# Add each stratum's effect as a point with confidence interval
y_positions = list(range(len(strata_names)))
coefficients = [stratum_results[stratum]['coefficient'] for stratum in strata_names]
std_errs = [stratum_results[stratum]['std_err'] for stratum in strata_names]
conf_lower = [stratum_results[stratum]['conf_int'][0] for stratum in strata_names]
conf_upper = [stratum_results[stratum]['conf_int'][1] for stratum in strata_names]

# Add horizontal line at zero
fig.add_shape(
    type="line",
    x0=0, y0=-0.5, x1=0, y1=len(strata_names)-0.5,
    line=dict(color="gray", width=1, dash="dash")
)

# Add points for effect sizes
fig.add_trace(
    go.Scatter(
        x=coefficients,
        y=strata_names,
        mode='markers',
        marker=dict(
            size=12,
            color='blue'
        ),
        name='Effect Size'
    )
)

# Add error bars (confidence intervals)
for i, stratum in enumerate(strata_names):
    fig.add_shape(
        type="line",
        x0=conf_lower[i], y0=i, x1=conf_upper[i], y1=i,
        line=dict(color="blue", width=2)
    )
    # Add caps to the CI lines
    for x in [conf_lower[i], conf_upper[i]]:
        fig.add_shape(
            type="line",
            x0=x, y0=i-0.1, x1=x, y1=i+0.1,
            line=dict(color="blue", width=2)
        )

# If confounding is included, also show adjusted results
if include_confounding:
    adj_coefficients = [stratum_results[stratum]['adj_coefficient'] for stratum in strata_names]
    adj_conf_lower = [stratum_results[stratum]['adj_conf_int'][0] for stratum in strata_names]
    adj_conf_upper = [stratum_results[stratum]['adj_conf_int'][1] for stratum in strata_names]
    
    # Add points for adjusted effect sizes
    fig.add_trace(
        go.Scatter(
            x=adj_coefficients,
            y=strata_names,
            mode='markers',
            marker=dict(
                size=12,
                color='red'
            ),
            name='Adjusted Effect Size'
        )
    )
    
    # Add error bars for adjusted CIs
    for i, stratum in enumerate(strata_names):
        fig.add_shape(
            type="line",
            x0=adj_conf_lower[i], y0=i-0.05, x1=adj_conf_upper[i], y1=i-0.05,
            line=dict(color="red", width=2)
        )
        # Add caps to the CI lines
        for x in [adj_conf_lower[i], adj_conf_upper[i]]:
            fig.add_shape(
                type="line",
                x0=x, y0=i-0.15, x1=x, y1=i+0.05,
                line=dict(color="red", width=2)
            )

# Update layout
fig.update_layout(
    title='Forest Plot: Effect Sizes by Stratum',
    xaxis_title=f'Effect of {exposure_name} on {outcome_name}',
    yaxis_title='Stratum',
    height=100 + 75 * len(strata_names),
    margin=dict(l=150)
)

# Store the figure in output_vars
output_vars['fig'] = fig

# Create a faceted scatter plot
fig2 = make_subplots(
    rows=1, 
    cols=len(strata_names), 
    shared_yaxes=True,
    subplot_titles=strata_names
)

# Color scale for the points
colors = px.colors.qualitative.Set1

# Add scatter plot for each stratum
for i, stratum in enumerate(strata_names):
    stratum_data = data[data['Stratum'] == stratum]
    
    # Add scatter points
    fig2.add_trace(
        go.Scatter(
            x=stratum_data[exposure_name],
            y=stratum_data[outcome_name],
            mode='markers',
            marker=dict(
                color=colors[i % len(colors)],
                size=8,
                opacity=0.6
            ),
            name=stratum
        ),
        row=1, col=i+1
    )
    
    # Add regression line
    x_range = np.linspace(
        stratum_data[exposure_name].min(),
        stratum_data[exposure_name].max(),
        100
    )
    
    coef = stratum_results[stratum]['coefficient']
    intercept = sm.OLS(
        stratum_data[outcome_name], 
        sm.add_constant(stratum_data[exposure_name])
    ).fit().params[0]
    
    y_pred = intercept + coef * x_range
    
    fig2.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(
                color=colors[i % len(colors)],
                width=3
            ),
            name=f'{stratum} (effect={coef:.2f})'
        ),
        row=1, col=i+1
    )

# Update layout
fig2.update_layout(
    title='Stratified Scatter Plots with Regression Lines',
    height=500,
    showlegend=False
)

# Update axis titles
for i in range(len(strata_names)):
    fig2.update_xaxes(title_text=exposure_name, row=1, col=i+1)
    
    if i == 0:
        fig2.update_yaxes(title_text=outcome_name, row=1, col=i+1)

# Store the second figure
output_vars['fig2'] = fig2
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and click Execute to see the results:", 
                           value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Change `stratify_by` to explore different scenarios
            - Modify `stratum_effects` to create different patterns of effect modification
            - Toggle `include_confounding` to see the impact of confounding
            - Adjust `sample_imbalance` to see how sample size differences affect results
            
            **Challenges:**
            1. Create a scenario where effect modification disappears after controlling for a confounder
            2. Make one stratum have a negative effect and others positive
            3. Increase sample imbalance and observe its impact on confidence intervals
            4. Set all stratum effects equal and confirm there's no significant effect modification
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Output:")
                st.text(output)
            
            # Display any figures generated
            if 'fig' in output_vars:
                st.plotly_chart(output_vars['fig'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Concepts")
    st.markdown("""
    ### Stratified Analysis for Effect Modification:
    
    1. **Approach to Stratification**:
       - Divide the population into subgroups based on the potential effect modifier
       - Analyze the exposure-outcome relationship separately within each stratum
       - Compare effect estimates across strata to assess heterogeneity
    
    2. **Confounding vs. Effect Modification**:
       - In confounding: Stratum-specific estimates differ from crude, but are similar to each other
       - In effect modification: Stratum-specific estimates differ from each other
       - Both can occur simultaneously in the same variable
    
    3. **Statistical Considerations**:
       - Power within each stratum depends on sample size
       - Imbalanced strata can lead to imprecise estimates in smaller groups
       - Tests for heterogeneity assess whether differences between strata are statistically significant
    
    4. **Visualization Methods**:
       - Forest plots display effect estimates and confidence intervals by stratum
       - Faceted scatter plots show the relationship within each stratum
       - Non-overlapping confidence intervals suggest effect modification
    
    5. **Measures to Consider**:
       - Absolute measures: Risk difference, rate difference
       - Relative measures: Risk ratio, rate ratio, odds ratio
       - Effect modification can be present on one scale but not another
    """)

def public_health_examples_lesson():
    st.header("Real-world Public Health Examples")
    
    st.markdown("""
    ### Effect Modification in Public Health Research
    
    Effect modification has important implications for public health research and practice.
    Understanding how interventions or exposures affect different subpopulations can help
    target resources more effectively and develop tailored prevention strategies.
    
    Let's explore some real-world public health scenarios:
    """)
    
    # Initial code example
    initial_code = """# Real-world public health effect modification examples
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
public_health_scenario = "vaccination"  # Options: "vaccination", "smoking", "diet_exercise", "screening"
n_samples = 1000  # Number of observations
effect_magnitude = 0.7  # Overall magnitude of effects (0-1)
noise_level = 0.5  # Amount of random variation

# Set random seed for reproducibility
np.random.seed(42)

# Configure variables based on selected scenario
if public_health_scenario == "vaccination":
    # Scenario: Vaccine efficacy varies by age group
    effect_modifier = "Age Group"
    modifier_categories = ["Child", "Adult", "Elderly"]
    exposure = "Vaccination"
    outcome = "Infection Risk"
    # Stronger protection in children and adults, weaker in elderly
    stratum_effects = [-0.8, -0.7, -0.3]  # Negative effects represent protection
    
elif public_health_scenario == "smoking":
    # Scenario: Smoking impact on lung cancer risk modified by genetic risk
    effect_modifier = "Genetic Risk"
    modifier_categories = ["Low", "Moderate", "High"]
    exposure = "Smoking (pack-years)"
    outcome = "Lung Cancer Risk"
    # Stronger effect in high genetic risk individuals
    stratum_effects = [0.4, 0.7, 1.0]
    
elif public_health_scenario == "diet_exercise":
    # Scenario: Combined effect of diet and exercise on weight loss
    effect_modifier = "Baseline BMI"
    modifier_categories = ["Normal", "Overweight", "Obese"]
    exposure = "Intervention Adherence"
    outcome = "Weight Loss (kg)"
    # Higher weight loss in those with higher baseline BMI
    stratum_effects = [0.3, 0.6, 0.9]
    
else:  # screening
    # Scenario: Cancer screening effectiveness by access to follow-up care
    effect_modifier = "Healthcare Access"
    modifier_categories = ["Low", "Medium", "High"]
    exposure = "Screening Participation"
    outcome = "Mortality Reduction"
    # Screening more effective with better healthcare access
    stratum_effects = [0.2, 0.5, 0.8]

# Scale effects by the overall magnitude
stratum_effects = [effect * effect_magnitude for effect in stratum_effects]

# Generate data
modifier_categories = modifier_categories[:3]  # Ensure we have at most 3 categories
stratum_effects = stratum_effects[:len(modifier_categories)]  # Match effects to categories

# Generate modifier variable (stratification variable)
modifier = np.random.choice(modifier_categories, n_samples)

# Generate exposure (could be binary or continuous depending on scenario)
if public_health_scenario == "vaccination":
    # Binary exposure for vaccination (0 = unvaccinated, 1 = vaccinated)
    exposure_data = np.random.binomial(1, 0.6, n_samples)
else:
    # Continuous exposure for other scenarios
    exposure_data = np.random.normal(0, 1, n_samples)

# Generate outcome with different effects by stratum
outcome_data = np.zeros(n_samples)

# Apply different effects based on modifier category
for i, category in enumerate(modifier_categories):
    mask = (modifier == category)
    effect = stratum_effects[i]
    
    # Generate outcome with category-specific effect
    outcome_data[mask] = exposure_data[mask] * effect + np.random.normal(0, noise_level, sum(mask))

# Create DataFrame
data = pd.DataFrame({
    effect_modifier: modifier,
    exposure: exposure_data,
    outcome: outcome_data
})

# Print data summary
print(f"Public Health Scenario: {public_health_scenario}")
print(f"Effect Modifier: {effect_modifier}")
print(f"Exposure: {exposure}")
print(f"Outcome: {outcome}")
print("\\nSample size by group:")
print(data[effect_modifier].value_counts())

# Perform stratified analysis
print("\\nStratified Analysis Results:")
print("-" * 50)

# Store results for visualization
results = {}

for category in modifier_categories:
    # Filter data for this category
    category_data = data[data[effect_modifier] == category]
    
    # Linear regression model
    X = sm.add_constant(category_data[exposure])
    model = sm.OLS(category_data[outcome], X).fit()
    
    # Store results
    results[category] = {
        'coefficient': model.params[1],
        'std_err': model.bse[1],
        'p_value': model.pvalues[1],
        'conf_int': model.conf_int().iloc[1].tolist(),
        'n': len(category_data)
    }
    
    # Print results
    print(f"Results for {category} {effect_modifier}:")
    print(f"  Sample size: {len(category_data)}")
    print(f"  Effect size: {model.params[1]:.3f}")
    print(f"  95% CI: [{model.conf_int().iloc[1, 0]:.3f}, {model.conf_int().iloc[1, 1]:.3f}]")
    print(f"  p-value: {model.pvalues[1]:.4f}")
    print("-" * 50)

# Test for effect modification using an interaction model
# Create dummy variables for categories (first category is reference)
for i, category in enumerate(modifier_categories[1:], 1):
    data[f'is_{category}'] = (data[effect_modifier] == category).astype(int)
    
    # Create interaction terms
    data[f'{exposure}_{category}'] = data[exposure] * data[f'is_{category}']

# Variables for the interaction model
interaction_vars = [f'{exposure}_{category}' for category in modifier_categories[1:]]
category_vars = [f'is_{category}' for category in modifier_categories[1:]]

# Fit interaction model
X_interaction = sm.add_constant(data[[exposure] + category_vars + interaction_vars])
interaction_model = sm.OLS(data[outcome], X_interaction).fit()

# Check if any interaction is significant
interaction_p_values = [interaction_model.pvalues[var] for var in interaction_vars]
significant_interaction = any(p < 0.05 for p in interaction_p_values)

print("Test for effect modification:")
for i, category in enumerate(modifier_categories[1:]):
    reference = modifier_categories[0]
    p_value = interaction_p_values[i]
    print(f"  {category} vs {reference}: p = {p_value:.4f}")

if significant_interaction:
    print("✓ Evidence of significant effect modification")
else:
    print("✗ No significant evidence of effect modification")

# Create a forest plot to visualize effect sizes across strata
fig = go.Figure()

# Add vertical line at zero
fig.add_shape(
    type="line",
    x0=0, y0=-0.5, x1=0, y1=len(modifier_categories)-0.5,
    line=dict(color="gray", width=1, dash="dash")
)

# Extract result values
coefficients = [results[category]['coefficient'] for category in modifier_categories]
conf_lower = [results[category]['conf_int'][0] for category in modifier_categories]
conf_upper = [results[category]['conf_int'][1] for category in modifier_categories]

# Add points for effect sizes
fig.add_trace(
    go.Scatter(
        x=coefficients,
        y=modifier_categories,
        mode='markers',
        marker=dict(
            size=14,
            color='blue'
        ),
        name='Effect Size'
    )
)

# Add error bars (confidence intervals)
for i, category in enumerate(modifier_categories):
    fig.add_shape(
        type="line",
        x0=conf_lower[i], y0=i, x1=conf_upper[i], y1=i,
        line=dict(color="blue", width=2)
    )
    # Add caps to the CI lines
    for x in [conf_lower[i], conf_upper[i]]:
        fig.add_shape(
            type="line",
            x0=x, y0=i-0.1, x1=x, y1=i+0.1,
            line=dict(color="blue", width=2)
        )

# Update layout
fig.update_layout(
    title=f'Effect of {exposure} on {outcome} by {effect_modifier}',
    xaxis_title=f'Effect Size (Regression Coefficient)',
    yaxis_title=effect_modifier,
    height=400,
    margin=dict(l=150)
)

# Store the figure in output_vars
output_vars['fig'] = fig

# Create data visualization appropriate to the scenario
if public_health_scenario == "vaccination" and exposure_data.dtype == bool:
    # For binary exposure like vaccination, create a bar chart of outcomes by group
    summary_data = []
    
    for category in modifier_categories:
        # Filter data for this category
        cat_data = data[data[effect_modifier] == category]
        
        # Calculate mean outcome for vaccinated and unvaccinated
        vacc_outcome = cat_data[cat_data[exposure] == 1][outcome].mean()
        unvacc_outcome = cat_data[cat_data[exposure] == 0][outcome].mean()
        
        # Calculate confidence intervals (assuming normal distribution)
        vacc_se = cat_data[cat_data[exposure] == 1][outcome].std() / np.sqrt(sum(cat_data[exposure] == 1))
        unvacc_se = cat_data[cat_data[exposure] == 0][outcome].std() / np.sqrt(sum(cat_data[exposure] == 0))
        
        vacc_ci = [vacc_outcome - 1.96 * vacc_se, vacc_outcome + 1.96 * vacc_se]
        unvacc_ci = [unvacc_outcome - 1.96 * unvacc_se, unvacc_outcome + 1.96 * unvacc_se]
        
        # Store results
        summary_data.append({
            effect_modifier: category,
            'Status': 'Vaccinated',
            outcome: vacc_outcome,
            'CI_Lower': vacc_ci[0],
            'CI_Upper': vacc_ci[1]
        })
        
        summary_data.append({
            effect_modifier: category,
            'Status': 'Unvaccinated',
            outcome: unvacc_outcome,
            'CI_Lower': unvacc_ci[0],
            'CI_Upper': unvacc_ci[1]
        })
    
    # Create DataFrame from summary data
    summary_df = pd.DataFrame(summary_data)
    
    # Create grouped bar chart
    fig2 = px.bar(
        summary_df,
        x=effect_modifier,
        y=outcome,
        color='Status',
        barmode='group',
        error_y='CI_Upper',
        error_y_minus='CI_Lower',
        title=f'{outcome} by {effect_modifier} and Vaccination Status',
        color_discrete_map={'Vaccinated': 'green', 'Unvaccinated': 'red'}
    )
    
    # Calculate vaccine efficacy for each group
    efficacy_data = []
    for category in modifier_categories:
        cat_summary = summary_df[summary_df[effect_modifier] == category]
        unvacc = cat_summary[cat_summary['Status'] == 'Unvaccinated'][outcome].values[0]
        vacc = cat_summary[cat_summary['Status'] == 'Vaccinated'][outcome].values[0]
        
        # Vaccine efficacy formula: (unvaccinated risk - vaccinated risk) / unvaccinated risk
        efficacy = (unvacc - vacc) / unvacc * 100
        
        efficacy_data.append({
            effect_modifier: category,
            'Vaccine Efficacy (%)': efficacy
        })
    
    # Create efficacy DataFrame
    efficacy_df = pd.DataFrame(efficacy_data)
    
    # Add text annotations
    for i, category in enumerate(modifier_categories):
        efficacy = efficacy_df[efficacy_df[effect_modifier] == category]['Vaccine Efficacy (%)'].values[0]
        
        fig2.add_annotation(
            x=category,
            y=summary_df[(summary_df[effect_modifier] == category) & 
                         (summary_df['Status'] == 'Unvaccinated')][outcome].values[0],
            text=f"Efficacy: {efficacy:.1f}%",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
else:
    # For continuous exposure, create faceted scatter plots
    fig2 = make_subplots(
        rows=1, 
        cols=len(modifier_categories), 
        shared_yaxes=True,
        subplot_titles=modifier_categories
    )
    
    # Color scale for the points
    colors = px.colors.qualitative.Set1
    
    # Add scatter plot for each category
    for i, category in enumerate(modifier_categories):
        category_data = data[data[effect_modifier] == category]
        
        # Add scatter points
        fig2.add_trace(
            go.Scatter(
                x=category_data[exposure],
                y=category_data[outcome],
                mode='markers',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8,
                    opacity=0.6
                ),
                name=category
            ),
            row=1, col=i+1
        )
        
        # Add regression line
        x_range = np.linspace(
            category_data[exposure].min(),
            category_data[exposure].max(),
            100
        )
        
        coef = results[category]['coefficient']
        intercept = sm.OLS(
            category_data[outcome], 
            sm.add_constant(category_data[exposure])
        ).fit().params[0]
        
        y_pred = intercept + coef * x_range
        
        fig2.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                line=dict(
                    color=colors[i % len(colors)],
                    width=3
                ),
                name=f'{category} (effect={coef:.2f})'
            ),
            row=1, col=i+1
        )
    
    # Update layout
    fig2.update_layout(
        title=f'Effect of {exposure} on {outcome} Across {effect_modifier} Groups',
        height=500,
        showlegend=False
    )
    
    # Update axis titles
    for i in range(len(modifier_categories)):
        fig2.update_xaxes(title_text=exposure, row=1, col=i+1)
        
        if i == 0:
            fig2.update_yaxes(title_text=outcome, row=1, col=i+1)

# Store the second figure
output_vars['fig2'] = fig2

# Create a third visualization - showing public health implications
if public_health_scenario == "vaccination":
    # For vaccination: Show number needed to vaccinate (NNV) by group
    nnv_data = []
    
    for category in modifier_categories:
        # Get effect size and outbreak risk from our data
        effect_size = abs(results[category]['coefficient'])
        
        # Calculate NNV based on effect size (simplistic approximation)
        # NNV = 1 / absolute risk reduction
        # Higher effect = lower NNV
        nnv = 1 / (effect_size * 0.1)  # Scale for visualization
        
        nnv_data.append({
            effect_modifier: category,
            'Number Needed to Vaccinate': nnv
        })
    
    # Create DataFrame
    nnv_df = pd.DataFrame(nnv_data)
    
    # Create bar chart
    fig3 = px.bar(
        nnv_df,
        x=effect_modifier,
        y='Number Needed to Vaccinate',
        title='Number Needed to Vaccinate to Prevent One Case by Age Group',
        color=effect_modifier,
        text_auto='.1f'
    )
    
    # Update layout
    fig3.update_layout(
        height=400,
        showlegend=False
    )
    
elif public_health_scenario == "smoking":
    # For smoking: Show risk difference and population attributable fraction
    paf_data = []
    
    # Assume smoking prevalence varies by genetic risk group
    prevalence = {
        "Low": 0.2,
        "Moderate": 0.3,
        "High": 0.25
    }
    
    for category in modifier_categories:
        # Get effect size (simplified RR)
        effect_size = results[category]['coefficient']
        smoking_prev = prevalence.get(category, 0.25)
        
        # Calculate population attributable fraction (PAF)
        # PAF = prevalence * (RR - 1) / [1 + prevalence * (RR - 1)]
        rr = 1 + effect_size  # Approximating relative risk
        paf = smoking_prev * (rr - 1) / (1 + smoking_prev * (rr - 1))
        
        # Calculate absolute risk difference
        risk_diff = effect_size
        
        paf_data.append({
            effect_modifier: category,
            'Population Attributable Fraction': paf * 100,
            'Risk Difference': risk_diff
        })
    
    # Create DataFrame
    paf_df = pd.DataFrame(paf_data)
    
    # Create dual axis chart
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add PAF bars
    fig3.add_trace(
        go.Bar(
            x=paf_df[effect_modifier],
            y=paf_df['Population Attributable Fraction'],
            name='PAF (%)',
            marker_color='blue',
            text=paf_df['Population Attributable Fraction'].round(1),
            textposition='auto'
        ),
        secondary_y=False
    )
    
    # Add risk difference line
    fig3.add_trace(
        go.Scatter(
            x=paf_df[effect_modifier],
            y=paf_df['Risk Difference'],
            name='Risk Difference',
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=3, color='red')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig3.update_layout(
        title='Population Impact of Smoking by Genetic Risk Group',
        height=400,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Update axis labels
    fig3.update_yaxes(title_text='Population Attributable Fraction (%)', secondary_y=False)
    fig3.update_yaxes(title_text='Risk Difference', secondary_y=True)
    
else:
    # For other scenarios: Create intervention impact visualization
    impact_data = []
    
    for category in modifier_categories:
        effect_size = results[category]['coefficient']
        
        # Calculate NNT (number needed to treat)
        # Higher effect = lower NNT
        nnt = 1 / abs(effect_size)
        
        # Calculate benefit per 100 treated
        benefit_per_100 = effect_size * 100
        
        impact_data.append({
            effect_modifier: category,
            'Number Needed to Treat': nnt,
            'Benefit per 100 Treated': np.abs(benefit_per_100)
        })
    
    # Create DataFrame
    impact_df = pd.DataFrame(impact_data)
    
    # Create dual axis chart
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add NNT bars (lower is better)
    fig3.add_trace(
        go.Bar(
            x=impact_df[effect_modifier],
            y=impact_df['Number Needed to Treat'],
            name='Number Needed to Treat',
            marker_color='orange',
            text=impact_df['Number Needed to Treat'].round(1),
            textposition='auto'
        ),
        secondary_y=False
    )
    
    # Add benefit line
    fig3.add_trace(
        go.Scatter(
            x=impact_df[effect_modifier],
            y=impact_df['Benefit per 100 Treated'],
            name='Benefit per 100 Treated',
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=3, color='green')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig3.update_layout(
        title=f'Intervention Impact by {effect_modifier}',
        height=400,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Update axis labels
    fig3.update_yaxes(title_text='Number Needed to Treat', secondary_y=False)
    fig3.update_yaxes(title_text='Benefit per 100 Treated', secondary_y=True)

# Store the third figure
output_vars['fig3'] = fig3

# Print public health implications based on the scenario
print("\\nPublic Health Implications:")

if public_health_scenario == "vaccination":
    print("1. Vaccination strategies should be tailored by age group")
    print(f"2. Highest vaccine efficacy observed in {modifier_categories[np.argmin(coefficients)]}")
    print(f"3. Additional protection measures may be needed for {modifier_categories[np.argmax(coefficients)]}")
    print("4. Vaccine resources could be allocated based on efficacy and risk in each group")
    
elif public_health_scenario == "smoking":
    print("1. Smoking cessation programs should consider genetic risk profiles")
    print(f"2. Highest smoking impact observed in {modifier_categories[np.argmax(coefficients)]}")
    print(f"3. Targeted screening may be justified in high-risk groups")
    print("4. Risk communication should be tailored based on genetic risk level")
    
elif public_health_scenario == "diet_exercise":
    print("1. Weight loss interventions have varying effectiveness by baseline BMI")
    print(f"2. Most effective for {modifier_categories[np.argmax(coefficients)]}")
    print(f"3. Alternative approaches may be needed for {modifier_categories[np.argmin(coefficients)]}")
    print("4. Expectations and goals should be tailored to baseline BMI")
    
else:  # screening
    print("1. Screening programs should consider healthcare access")
    print(f"2. Greatest benefit observed in {modifier_categories[np.argmax(coefficients)]}")
    print("3. For lower access groups, screening must be coupled with improved follow-up care")
    print("4. Resources to improve healthcare access could enhance screening effectiveness")
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and click Execute to see the results:", 
                           value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Change `public_health_scenario` to explore different public health contexts
            - Adjust `effect_magnitude` to see how it affects the strength of effect modification
            - Modify `noise_level` to see how random variation affects detection of effects
            - Try different values for `n_samples` to see how sample size affects results
            
            **Challenges:**
            1. Create a scenario where one group shows no effect (coefficient near zero)
            2. Find the minimum sample size needed for statistically significant results
            3. Increase noise level until effect modification is no longer detectable
            4. Modify the code to add a fourth category to one of the scenarios
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Output:")
                st.text(output)
            
            # Display any figures generated
            if 'fig' in output_vars:
                st.plotly_chart(output_vars['fig'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
                
            if 'fig3' in output_vars:
                st.plotly_chart(output_vars['fig3'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Concepts")
    st.markdown("""
    ### Public Health Applications of Effect Modification:
    
    1. **Vaccination Programs**:
       - Vaccine efficacy varies by age, comorbidities, and genetics
       - Understanding effect modification helps optimize vaccination strategies
       - Resource allocation can be guided by stratum-specific effectiveness
       - Risk communication should be tailored to different groups
    
    2. **Risk Factor Interventions**:
       - Impact of risk factors (smoking, diet, etc.) varies across populations
       - Intervention programs can be targeted to groups with largest potential benefit
       - Prevention messaging can be customized based on susceptibility
       - Population attributable fractions may vary by subgroups
    
    3. **Screening Programs**:
       - Effectiveness of screening depends on baseline risk and access to care
       - Cost-effectiveness calculations should incorporate effect modification
       - Screening intervals might be tailored based on effect modifiers
       - Different screening modalities might be appropriate for different groups
    
    4. **Policy Implications**:
       - One-size-fits-all policies may not be optimal
       - Effect modification can inform more nuanced, targeted approaches
       - Health disparities may reflect effect modification by social determinants
       - Resource allocation decisions should consider heterogeneity of effects
    
    5. **Research Considerations**:
       - Studies should be designed with sufficient power to detect effect modification
       - Important effect modifiers should be identified a priori
       - Stratified randomization may be appropriate in clinical trials
       - Pooled results may mask important differences between subgroups
    """)

if __name__ == "__main__":
    app()