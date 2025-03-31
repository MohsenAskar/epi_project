import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from contextlib import redirect_stdout
import sys
import traceback

def app():
    st.title("Interactive Coding Laboratory: Selection Bias Analysis")
    
    st.markdown("""
    ## Learn by Coding: Selection Bias Concepts
    
    This interactive coding laboratory allows you to modify and execute Python code directly in your browser.
    Experiment with different types of selection bias by modifying the example code and observing the results.
    
    Choose a topic to explore:
    """)
    
    # Topic selection - simplified to just 2 options for first-year students
    topic = st.selectbox(
        "Select a selection bias topic:",
        ["Basic Selection Bias Simulation", 
         "Effect of Selection Bias on Study Results"]
    )
    
    # Display the selected topic
    if topic == "Basic Selection Bias Simulation":
        basic_selection_bias_lesson()
    elif topic == "Effect of Selection Bias on Study Results":
        selection_bias_effects_lesson()

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
                'px': px,
                'go': go,
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

def basic_selection_bias_lesson():
    st.header("Basic Selection Bias Simulation")
    
    st.markdown("""
    ### Understanding Selection Bias
    
    Selection bias occurs when the individuals included in a study differ systematically from those 
    who were not included, and this difference is related to both the exposure and outcome.
    
    Let's create a simple simulation to demonstrate how selection bias works:
    """)
    
    # Initial code example - simplified for beginners
    initial_code = """# Basic selection bias simulation
import numpy as np
import pandas as pd
import plotly.express as px

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
population_size = 1000       # Total number of individuals
selection_strength = 0.7     # How strong the selection bias is (0-1)
exposure_rate = 0.4          # Proportion of exposed individuals
true_effect = 0.3            # True effect of exposure on outcome
bias_type = "outcome"        # Type of bias: "outcome", "exposure", or "both"

# Set random seed for reproducibility
np.random.seed(42)

# Generate the exposure variable (0 = not exposed, 1 = exposed)
exposure = np.random.binomial(1, exposure_rate, population_size)

# Generate the outcome based on exposure
# Baseline risk plus the effect of exposure
outcome_probability = 0.2 + (true_effect * exposure)
outcome = np.random.binomial(1, outcome_probability, population_size)

# Generate selection probability based on bias type
if bias_type == "outcome":
    # Outcome-dependent selection: people with the outcome are more likely to be selected
    selection_probability = 0.3 + (selection_strength * outcome)
    bias_description = "Outcome-dependent selection"
    
elif bias_type == "exposure":
    # Exposure-dependent selection: exposed people are more likely to be selected
    selection_probability = 0.3 + (selection_strength * exposure)
    bias_description = "Exposure-dependent selection"
    
elif bias_type == "both":
    # Selection depends on both exposure and outcome
    selection_probability = 0.3 + (selection_strength * (exposure + outcome) / 2)
    bias_description = "Selection dependent on both exposure and outcome"
    
else:
    # No bias (random selection)
    selection_probability = 0.5 * np.ones(population_size)
    bias_description = "Random selection (no bias)"

# Ensure probability is between 0 and 1
selection_probability = np.clip(selection_probability, 0, 1)

# Generate selection based on probability
selected = np.random.binomial(1, selection_probability, population_size)

# Create DataFrame
data = pd.DataFrame({
    'Exposure': exposure,
    'Outcome': outcome,
    'Selection_Probability': selection_probability,
    'Selected': selected
})

# Calculate the true effect in the full population
# Count number in each exposure-outcome group
exposed_with_outcome = np.sum((data['Exposure'] == 1) & (data['Outcome'] == 1))
exposed_without_outcome = np.sum((data['Exposure'] == 1) & (data['Outcome'] == 0))
unexposed_with_outcome = np.sum((data['Exposure'] == 0) & (data['Outcome'] == 1))
unexposed_without_outcome = np.sum((data['Exposure'] == 0) & (data['Outcome'] == 0))

# Calculate risk in exposed and unexposed groups
risk_exposed = exposed_with_outcome / (exposed_with_outcome + exposed_without_outcome)
risk_unexposed = unexposed_with_outcome / (unexposed_with_outcome + unexposed_without_outcome)

# Calculate risk ratio in full population
true_risk_ratio = risk_exposed / risk_unexposed

# Now calculate the same measures but only in the selected group
selected_data = data[data['Selected'] == 1]

# Count number in each exposure-outcome group for selected participants
sel_exposed_with_outcome = np.sum((selected_data['Exposure'] == 1) & (selected_data['Outcome'] == 1))
sel_exposed_without_outcome = np.sum((selected_data['Exposure'] == 1) & (selected_data['Outcome'] == 0))
sel_unexposed_with_outcome = np.sum((selected_data['Exposure'] == 0) & (selected_data['Outcome'] == 1))
sel_unexposed_without_outcome = np.sum((selected_data['Exposure'] == 0) & (selected_data['Outcome'] == 0))

# Calculate risk in exposed and unexposed groups for selected participants
sel_risk_exposed = sel_exposed_with_outcome / (sel_exposed_with_outcome + sel_exposed_without_outcome)
sel_risk_unexposed = sel_unexposed_with_outcome / (sel_unexposed_with_outcome + sel_unexposed_without_outcome)

# Calculate risk ratio in selected population
observed_risk_ratio = sel_risk_exposed / sel_risk_unexposed

# Print the results
print(f"Selection Bias Type: {bias_description}")
print(f"Selection Strength: {selection_strength}")
print(f"True Effect (Risk Ratio) in Full Population: {true_risk_ratio:.2f}")
print(f"Observed Effect (Risk Ratio) in Selected Sample: {observed_risk_ratio:.2f}")

if abs(observed_risk_ratio - true_risk_ratio) > 0.1:
    percent_diff = ((observed_risk_ratio - true_risk_ratio) / true_risk_ratio) * 100
    print(f"BIAS DETECTED: The observed risk ratio differs from the true risk ratio by {percent_diff:.1f}%")
else:
    print("No substantial bias detected in this simulation")

print("\\nFull Population Size:", population_size)
print("Selected Sample Size:", len(selected_data))
print(f"Selection Rate: {(len(selected_data)/population_size)*100:.1f}%")

# Create a contingency table of selection by outcome status
selection_by_outcome = pd.crosstab(
    data['Outcome'], 
    data['Selected'],
    normalize='index'
) * 100  # Convert to percentage

print("\\nPercentage Selected by Outcome Status:")
print(selection_by_outcome.round(1))

# Create a contingency table of selection by exposure status
selection_by_exposure = pd.crosstab(
    data['Exposure'], 
    data['Selected'],
    normalize='index'
) * 100  # Convert to percentage

print("\\nPercentage Selected by Exposure Status:")
print(selection_by_exposure.round(1))

# Create a scatter plot showing who gets selected
# Add a small amount of random jitter to visualize the binary data better
jittered_data = data.copy()
jittered_data['Exposure'] = jittered_data['Exposure'] + np.random.uniform(-0.1, 0.1, len(jittered_data))
jittered_data['Outcome'] = jittered_data['Outcome'] + np.random.uniform(-0.1, 0.1, len(jittered_data))

fig = px.scatter(
    jittered_data, 
    x='Exposure', 
    y='Outcome',
    color='Selected', 
    color_discrete_map={0: "lightgrey", 1: "red"},
    title="Selection Patterns by Exposure and Outcome Status",
    labels={"Exposure": "Exposure (1=Yes, 0=No)", "Outcome": "Outcome (1=Yes, 0=No)"},
    height=500
)

# Set axis ranges and ticks
fig.update_xaxes(range=[-0.2, 1.2], tickvals=[0, 1], title_text="Exposure Status")
fig.update_yaxes(range=[-0.2, 1.2], tickvals=[0, 1], title_text="Outcome Status")

# Add a legend
fig.update_layout(
    legend_title="Selection Status",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Store the figure in output_vars to display it
output_vars['fig'] = fig

# Create a bar chart comparing true vs observed risk ratio
fig2 = go.Figure()

# Add bars for the risk ratios
fig2.add_trace(
    go.Bar(
        x=['True Risk Ratio', 'Observed Risk Ratio'],
        y=[true_risk_ratio, observed_risk_ratio],
        text=[f'{true_risk_ratio:.2f}', f'{observed_risk_ratio:.2f}'],
        textposition='auto',
        marker_color=['blue', 'red']
    )
)

# Add a reference line at y=1 (no effect)
fig2.add_shape(
    type="line",
    x0=-0.5, y0=1, x1=1.5, y1=1,
    line=dict(color="black", width=1, dash="dash")
)

# Update layout
fig2.update_layout(
    title='Impact of Selection Bias on Risk Ratio',
    yaxis_title='Risk Ratio',
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
        # Add hints and challenges - simplified for beginners
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Change `bias_type` to "exposure", "outcome", or "both" to see different types of selection bias
            - Adjust `selection_strength` to see how the strength of bias affects the results
            - Try `true_effect = 0` to see if selection bias can create an association when none exists
            - Increase or decrease `population_size` to see how that affects the stability of results
            
            **Simple Challenges:**
            1. Find which `bias_type` creates the largest distortion in the risk ratio
            2. Try setting `selection_strength = 0` and see if there's any bias
            3. What happens if `exposure_rate = 0.8` (most people are exposed)?
            4. Can you create a situation where the observed risk ratio is less than 1 when the true risk ratio is greater than 1?
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
    
    # Include a discussion section - simplified for beginners
    st.subheader("Key Concepts")
    st.markdown("""
    ### Understanding Selection Bias:
    
    1. **What is Selection Bias?**
       - Selection bias occurs when the relationship between exposure and outcome is different 
         in your study sample compared to the target population
       - It happens when the selection process is related to both the exposure and the outcome
    
    2. **Common Types of Selection Bias:**
       - **Outcome-dependent selection**: People with the outcome are more likely to participate
         (e.g., people with symptoms more likely to join a health study)
       - **Exposure-dependent selection**: Exposed individuals are more likely to participate
         (e.g., only including current smokers in a smoking study)
       - **Both-dependent selection**: Selection depends on both exposure and outcome
         (e.g., hospital-based studies where both exposure and outcome affect hospitalization)
    
    3. **Effects of Selection Bias:**
       - Can create an association when none truly exists
       - Can make a true association appear stronger or weaker
       - Can even reverse the direction of an association (protective factor appears harmful)
    
    4. **Why It Matters:**
       - Selection bias can lead to incorrect conclusions about cause and effect
       - It cannot be fixed by simply increasing sample size
       - Understanding selection bias helps design better studies and interpret results correctly
    """)

def selection_bias_effects_lesson():
    st.header("Effect of Selection Bias on Study Results")
    
    st.markdown("""
    ### How Selection Bias Affects Study Results
    
    Different types of selection bias can affect study results in different ways. In this exercise,
    we'll simulate a hypothetical epidemiological study and see how various selection mechanisms
    can distort the observed relationship between exposure and outcome.
    """)
    
    # Initial code example - simplified for beginners
    initial_code = """# Simulating different types of selection bias
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
population_size = 2000       # Total population size
true_effect = 0.3            # True effect of exposure on outcome (risk difference)
selection_scenario = "case_control"  # Options: "case_control", "loss_to_followup", "healthy_worker"
selection_strength = 0.5     # Strength of selection bias (0-1)

# Set random seed for reproducibility
np.random.seed(42)

# Generate baseline characteristics
age = np.random.normal(45, 15, population_size)  # Age (normally distributed)
male = np.random.binomial(1, 0.5, population_size)  # Sex (1=male, 0=female)

# Generate exposure based on the scenario
if selection_scenario == "case_control":
    # Simple random exposure
    exposure = np.random.binomial(1, 0.4, population_size)
    
elif selection_scenario == "loss_to_followup":
    # Exposure partly determined by age (older people more likely to be exposed)
    p_exposure = 1 / (1 + np.exp(-(age - 45) / 10))  # Logistic function
    exposure = np.random.binomial(1, p_exposure, population_size)
    
elif selection_scenario == "healthy_worker":
    # Exposure (being employed) related to health status
    health = np.random.normal(0, 1, population_size)  # Underlying health
    p_exposure = 1 / (1 + np.exp(-health))  # Healthier people more likely to be employed
    exposure = np.random.binomial(1, p_exposure, population_size)
    
else:  # Default case
    exposure = np.random.binomial(1, 0.5, population_size)

# Generate outcome based on exposure and scenario
if selection_scenario == "case_control" or selection_scenario == "loss_to_followup":
    # Baseline risk plus effect of exposure
    p_outcome = 0.2 + (true_effect * exposure)
    
elif selection_scenario == "healthy_worker":
    # Outcome depends on exposure and underlying health
    # Note: In the healthy worker effect, employment (exposure) is actually protective
    # but this association is partly due to healthier people being more likely to be employed
    health_effect = -0.3 * health  # Negative effect (better health = lower disease risk)
    p_outcome = 0.3 + (true_effect * exposure) + health_effect
    
else:  # Default case
    p_outcome = 0.2 + (true_effect * exposure)

# Ensure probability is between 0 and 1
p_outcome = np.clip(p_outcome, 0, 1)
outcome = np.random.binomial(1, p_outcome, population_size)

# Generate selection probability based on scenario
if selection_scenario == "case_control":
    # In a case-control study, all cases are selected and only a sample of controls
    selection_probability = (outcome == 1) | (np.random.random(population_size) < 0.2)
    selection_probability = selection_probability.astype(float)
    bias_description = "Case-Control Selection"
    
elif selection_scenario == "loss_to_followup":
    # People with both exposure and outcome more likely to be lost (not selected)
    # Higher selection strength means more differential loss
    selection_probability = 0.9 - (selection_strength * exposure * outcome)
    bias_description = "Differential Loss to Follow-up"
    
elif selection_scenario == "healthy_worker":
    # Selection into employment study related to health status
    # Healthier people more likely to be included (selected)
    selection_probability = 0.3 + (selection_strength * health)
    bias_description = "Healthy Worker Effect"
    
else:  # Default: random selection
    selection_probability = 0.5 * np.ones(population_size)
    bias_description = "Random Selection (No Bias)"

# Ensure probability is between 0 and 1
selection_probability = np.clip(selection_probability, 0, 1)

# Generate selection based on probability
selected = np.random.binomial(1, selection_probability, population_size)

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Male': male,
    'Exposure': exposure,
    'Outcome': outcome,
    'Selection_Probability': selection_probability,
    'Selected': selected
})

# Calculate the true effect in the full population
# Use risk difference (easier for beginners to understand)
total_exposed = sum(data['Exposure'] == 1)
total_unexposed = sum(data['Exposure'] == 0)
outcome_in_exposed = sum((data['Exposure'] == 1) & (data['Outcome'] == 1))
outcome_in_unexposed = sum((data['Exposure'] == 0) & (data['Outcome'] == 1))

risk_exposed = outcome_in_exposed / total_exposed
risk_unexposed = outcome_in_unexposed / total_unexposed
true_risk_difference = risk_exposed - risk_unexposed

# Now calculate the same measures but only in the selected group
selected_data = data[data['Selected'] == 1]

# Calculate measures in selected sample
sel_total_exposed = sum(selected_data['Exposure'] == 1)
sel_total_unexposed = sum(selected_data['Exposure'] == 0)
sel_outcome_in_exposed = sum((selected_data['Exposure'] == 1) & (selected_data['Outcome'] == 1))
sel_outcome_in_unexposed = sum((selected_data['Exposure'] == 0) & (selected_data['Outcome'] == 1))

sel_risk_exposed = sel_outcome_in_exposed / sel_total_exposed
sel_risk_unexposed = sel_outcome_in_unexposed / sel_total_unexposed
observed_risk_difference = sel_risk_exposed - sel_risk_unexposed

# Print results
print(f"Selection Bias Scenario: {bias_description}")
print(f"Selection Strength: {selection_strength}")
print(f"\\nTrue Population Size: {population_size}")
print(f"Selected Sample Size: {len(selected_data)}")
print(f"Selection Rate: {len(selected_data)/population_size:.1%}")

print(f"\\nTrue Risk in Exposed Group: {risk_exposed:.1%}")
print(f"True Risk in Unexposed Group: {risk_unexposed:.1%}")
print(f"True Risk Difference: {true_risk_difference:.1%}")

print(f"\\nObserved Risk in Exposed Group: {sel_risk_exposed:.1%}")
print(f"Observed Risk in Unexposed Group: {sel_risk_unexposed:.1%}")
print(f"Observed Risk Difference: {observed_risk_difference:.1%}")

# Calculate the magnitude and direction of bias
bias_amount = observed_risk_difference - true_risk_difference
bias_percent = (bias_amount / true_risk_difference) * 100 if true_risk_difference != 0 else float('inf')

print(f"\\nBias Amount: {bias_amount:.1%}")
print(f"Bias Percent: {bias_percent:.1f}%")

if abs(bias_percent) > 10:  # More than 10% bias
    if bias_amount > 0:
        print("BIAS DETECTED: The study overestimates the true effect.")
    else:
        print("BIAS DETECTED: The study underestimates the true effect.")
else:
    print("No substantial bias detected in this simulation.")

# Create visualizations
# 1. Bar chart comparing true vs. observed risk difference
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=['True Effect', 'Observed Effect'],
        y=[true_risk_difference, observed_risk_difference],
        text=[f'{true_risk_difference:.1%}', f'{observed_risk_difference:.1%}'],
        textposition='auto',
        marker_color=['blue', 'red']
    )
)

# Add a reference line at 0 (no effect)
fig.add_shape(
    type="line",
    x0=-0.5, y0=0, x1=1.5, y1=0,
    line=dict(color="black", width=1, dash="dash")
)

fig.update_layout(
    title='Impact of Selection Bias on Risk Difference',
    yaxis_title='Risk Difference',
    height=400
)

output_vars['fig'] = fig

# 2. Visualization showing selection patterns
if selection_scenario == "case_control":
    # For case-control: Show selection rate by outcome status
    selection_by_outcome = pd.crosstab(
        data['Outcome'],
        data['Selected'],
        normalize='index'
    ) * 100
    
    fig2 = px.bar(
        x=['No Disease', 'Disease'],
        y=[selection_by_outcome.loc[0, 1], selection_by_outcome.loc[1, 1]],
        labels={'x': 'Disease Status', 'y': 'Selection Percentage'},
        title='Selection Rate by Disease Status in Case-Control Study',
        text_auto='.1f'
    )
    
    fig2.update_layout(height=400, yaxis_range=[0, 100])
    
elif selection_scenario == "loss_to_followup":
    # For loss to follow-up: Show selection rate by exposure-outcome combination
    # Create a new column for the combination
    data['Group'] = 'Unexposed, No Outcome'
    data.loc[(data['Exposure'] == 1) & (data['Outcome'] == 0), 'Group'] = 'Exposed, No Outcome'
    data.loc[(data['Exposure'] == 0) & (data['Outcome'] == 1), 'Group'] = 'Unexposed, Outcome'
    data.loc[(data['Exposure'] == 1) & (data['Outcome'] == 1), 'Group'] = 'Exposed, Outcome'
    
    # Calculate selection rates by group
    selection_by_group = data.groupby('Group')['Selected'].mean() * 100
    
    # Order the groups logically
    ordered_groups = ['Unexposed, No Outcome', 'Exposed, No Outcome', 
                      'Unexposed, Outcome', 'Exposed, Outcome']
    selection_by_group = selection_by_group.reindex(ordered_groups)
    
    fig2 = px.bar(
        x=selection_by_group.index,
        y=selection_by_group.values,
        labels={'x': 'Exposure-Outcome Group', 'y': 'Selection Percentage'},
        title='Selection Rate by Exposure-Outcome Group (Loss to Follow-up)',
        text_auto='.1f'
    )
    
    fig2.update_layout(height=400, yaxis_range=[0, 100])
    
elif selection_scenario == "healthy_worker":
    # For healthy worker: Show how health status affects both exposure and selection
    # Create health quartiles for better visualization
    data['Health_Quartile'] = pd.qcut(data['Age'], 4, labels=['Lowest', 'Low', 'High', 'Highest'])
    
    # Calculate exposure and selection rates by health quartile
    rates_by_health = data.groupby('Health_Quartile').agg({
        'Exposure': 'mean',
        'Selected': 'mean'
    }) * 100
    
    # Create a grouped bar chart
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Bar(
            x=rates_by_health.index,
            y=rates_by_health['Exposure'],
            name='Exposure Rate',
            marker_color='blue',
            text=[f'{val:.1f}%' for val in rates_by_health['Exposure']],
            textposition='auto'
        )
    )
    
    fig2.add_trace(
        go.Bar(
            x=rates_by_health.index,
            y=rates_by_health['Selected'],
            name='Selection Rate',
            marker_color='red',
            text=[f'{val:.1f}%' for val in rates_by_health['Selected']],
            textposition='auto'
        )
    )
    
    fig2.update_layout(
        title='Exposure and Selection Rates by Health Status',
        xaxis_title='Health Status',
        yaxis_title='Percentage',
        barmode='group',
        height=400,
        yaxis_range=[0, 100]
    )
    
else:
    # Default visualization
    fig2 = px.scatter(
        data,
        x='Exposure',
        y='Outcome',
        color='Selected',
        title='Selection Pattern by Exposure and Outcome',
        labels={'Exposure': 'Exposure Status', 'Outcome': 'Outcome Status'},
        color_discrete_map={0: 'lightgrey', 1: 'red'}
    )
    
    # Add jitter for better visualization
    fig2.update_traces(
        x=data['Exposure'] + np.random.uniform(-0.1, 0.1, len(data)),
        y=data['Outcome'] + np.random.uniform(-0.1, 0.1, len(data))
    )
    
    fig2.update_layout(height=400)

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
        # Add hints and challenges - simplified for beginners
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Change `selection_scenario` to "case_control", "loss_to_followup", or "healthy_worker"
            - Adjust `selection_strength` to see how it affects the amount of bias
            - Change `true_effect` to see how different effect sizes are affected by bias
            - Try increasing `population_size` to see if it reduces bias
            
            **Simple Challenges:**
            1. Which scenario creates the most bias with `selection_strength = 0.5`?
            2. In the "healthy_worker" scenario, try different values of `selection_strength` to find when the observed effect appears beneficial (negative) even though the true effect is harmful (positive)
            3. For "loss_to_followup", what happens to the bias as you increase `selection_strength`?
            4. Try `true_effect = 0` to see if any scenario creates the appearance of an effect when none exists
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
    
    # Include a discussion section - simplified for beginners
    st.subheader("Key Concepts")
    st.markdown("""
    ### Common Types of Selection Bias in Epidemiology:
    
    1. **Case-Control Selection Bias:**
       - In case-control studies, we select people based on whether they have the outcome
       - All or most cases (people with the outcome) are included
       - Only a sample of controls (people without the outcome) are included
       - This can distort exposure-outcome relationships if selection of controls is related to the exposure
    
    2. **Loss to Follow-up:**
       - Occurs in cohort studies when participants drop out non-randomly
       - If dropout is related to both exposure and outcome, bias occurs
       - Example: Participants experiencing side effects (exposure + outcome) drop out of a drug trial
       - Typically biases toward seeing smaller effects than truly exist
    
    3. **Healthy Worker Effect:**
       - A special form of selection bias in occupational studies
       - Employed people tend to be healthier than the general population
       - Creates an apparent protective effect of employment
       - Makes occupational exposures appear less harmful than they truly are
    
    4. **How Selection Bias Affects Results:**
       - Can make harmful exposures appear beneficial
       - Can make beneficial exposures appear harmful
       - Can create the illusion of an effect when none exists
       - Can hide true effects by biasing toward the null
    
    5. **Addressing Selection Bias:**
       - Careful study design to minimize differential selection
       - Collecting data on reasons for non-participation
       - Statistical methods like inverse probability weighting
       - Sensitivity analyses to assess the potential impact of bias
    """)

if __name__ == "__main__":
    app()