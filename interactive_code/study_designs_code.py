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
    st.title("Interactive Coding Laboratory: Epidemiological Study Designs")
    
    st.markdown("""
    ## Learn by Coding: Epidemiological Study Designs
    
    This interactive laboratory allows you to explore different epidemiological study designs 
    through code. Modify the example code and see how different design choices affect results.
    
    Choose a topic to explore:
    """)
    
    # Simplified topic selection for first-year students
    topic = st.selectbox(
        "Select a topic:",
        ["Basic Study Designs", 
         "Comparing Study Designs"]
    )
    
    # Display the selected topic
    if topic == "Basic Study Designs":
        basic_study_designs_lesson()
    elif topic == "Comparing Study Designs":
        comparing_study_designs_lesson()

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

def basic_study_designs_lesson():
    st.header("Basic Epidemiological Study Designs")
    
    st.markdown("""
    ### Understanding Different Study Designs
    
    Epidemiological studies use different designs to investigate relationships between exposures and outcomes.
    Each design has unique features, strengths, and limitations.
    
    Let's explore the most common study designs by simulating data for each:
    """)
    
    # Study design selector - simplified for first-year students
    study_design = st.selectbox(
        "Choose a study design to simulate:",
        ["Cohort Study", "Case-Control Study", "Cross-sectional Study", "Randomized Controlled Trial"]
    )
    
    # Display different code examples based on the selected study design
    if study_design == "Cohort Study":
        initial_code = """# Simulation of a Cohort Study
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
population_size = 1000     # Number of people in the study
exposure_rate = 0.4        # Proportion of population exposed (40%)
followup_years = 5         # Years of follow-up
baseline_risk = 0.05       # Annual risk of disease in unexposed group (5%)
relative_risk = 2.0        # Risk in exposed relative to unexposed

# Set random seed for reproducibility
np.random.seed(42)

# Generate exposure status for each person
exposed = np.random.binomial(1, exposure_rate, population_size)
exposed_count = sum(exposed)
unexposed_count = population_size - exposed_count

print(f"Study Population: {population_size} people")
print(f"Exposed: {exposed_count} people ({exposed_count/population_size:.1%})")
print(f"Unexposed: {unexposed_count} people ({unexposed_count/population_size:.1%})")

# Simulate disease occurrence
# For simplicity, we'll use a basic exponential time-to-event model
def generate_time_to_disease(baseline_risk, rr, n_people):
    # Generate time until disease event occurs
    # Convert annual risk to a rate parameter
    rate = -np.log(1 - baseline_risk)
    # Apply relative risk for exposed
    adjusted_rate = rate * rr
    # Generate random times from exponential distribution
    return np.random.exponential(1/adjusted_rate, n_people)

# Generate time to disease for unexposed and exposed
time_to_disease_unexposed = generate_time_to_disease(baseline_risk, 1.0, unexposed_count)
time_to_disease_exposed = generate_time_to_disease(baseline_risk, relative_risk, exposed_count)

# Create a DataFrame to store our cohort data
cohort_data = pd.DataFrame({
    'Exposure': ['Exposed'] * exposed_count + ['Unexposed'] * unexposed_count,
    'Time_to_Disease': np.concatenate([time_to_disease_exposed, time_to_disease_unexposed])
})

# Determine who gets disease within follow-up period
cohort_data['Disease'] = cohort_data['Time_to_Disease'] <= followup_years
cohort_data['Time_Observed'] = np.minimum(cohort_data['Time_to_Disease'], followup_years)

# Calculate results
exposed_cases = sum((cohort_data['Exposure'] == 'Exposed') & cohort_data['Disease'])
unexposed_cases = sum((cohort_data['Exposure'] == 'Unexposed') & cohort_data['Disease'])

# Calculate risks in each group
risk_exposed = exposed_cases / exposed_count
risk_unexposed = unexposed_cases / unexposed_count

# Calculate risk ratio and risk difference
risk_ratio = risk_exposed / risk_unexposed
risk_difference = risk_exposed - risk_unexposed

# Calculate person-time
person_years_exposed = cohort_data[cohort_data['Exposure'] == 'Exposed']['Time_Observed'].sum()
person_years_unexposed = cohort_data[cohort_data['Exposure'] == 'Unexposed']['Time_Observed'].sum()

# Calculate incidence rates
ir_exposed = exposed_cases / person_years_exposed
ir_unexposed = unexposed_cases / person_years_unexposed
ir_ratio = ir_exposed / ir_unexposed

# Print results
print("\\nCohort Study Results:")
print("-" * 40)
print(f"Follow-up period: {followup_years} years")
print(f"Disease in exposed: {exposed_cases} cases ({risk_exposed:.1%})")
print(f"Disease in unexposed: {unexposed_cases} cases ({risk_unexposed:.1%})")
print(f"Risk Ratio: {risk_ratio:.2f}")
print(f"Risk Difference: {risk_difference:.2%}")
print("\\nIncidence Rate Results:")
print(f"Incidence Rate (exposed): {ir_exposed:.3f} cases per person-year")
print(f"Incidence Rate (unexposed): {ir_unexposed:.3f} cases per person-year")
print(f"Rate Ratio: {ir_ratio:.2f}")

# Create contingency table
contingency = pd.crosstab(cohort_data['Exposure'], cohort_data['Disease'])
print("\\nContingency Table:")
print(contingency)

# VISUALIZATIONS

# 1. Bar chart of disease occurrence by exposure status
exposure_disease = cohort_data.groupby('Exposure')['Disease'].mean().reset_index()
exposure_disease['Disease_Percentage'] = exposure_disease['Disease'] * 100

fig1 = px.bar(
    exposure_disease,
    x='Exposure',
    y='Disease_Percentage',
    title='Disease Occurrence by Exposure Status',
    labels={'Disease_Percentage': 'Disease Occurrence (%)', 'Exposure': 'Exposure Status'},
    color='Exposure',
    text_auto='.1f'
)

fig1.update_layout(height=500)
output_vars['fig1'] = fig1

# 2. Survival curves by exposure status
# Create a time grid
time_grid = np.linspace(0, followup_years, 100)

# Calculate survival proportions at each time point
survival_data = []
for exposure in ['Exposed', 'Unexposed']:
    times = cohort_data[cohort_data['Exposure'] == exposure]['Time_to_Disease'].values
    for t in time_grid:
        survival_prop = np.mean(times > t)
        survival_data.append({
            'Time': t,
            'Survival': survival_prop,
            'Exposure': exposure
        })

survival_df = pd.DataFrame(survival_data)

fig2 = px.line(
    survival_df,
    x='Time',
    y='Survival',
    color='Exposure',
    title='Survival Curves by Exposure Status',
    labels={'Time': 'Time (years)', 'Survival': 'Proportion Without Disease'},
    line_shape='spline'  # smooth line
)

fig2.update_layout(height=500)
output_vars['fig2'] = fig2
"""
    elif study_design == "Case-Control Study":
        initial_code = """# Simulation of a Case-Control Study
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_cases = 200                # Number of disease cases
n_controls = 400             # Number of controls (people without disease)
true_odds_ratio = 2.5        # True association between exposure and disease
exposure_prevalence = 0.3    # Prevalence of exposure in the general population

# Set random seed for reproducibility
np.random.seed(42)

# First, imagine a hypothetical source population
# We'll work backwards from our desired odds ratio

# In the general population, exposure prevalence is 30%
# We need to calculate exposure prevalence in cases, given our target odds ratio
# Using the odds ratio formula: OR = (a/c) / (b/d)
# Where a = exposed cases, b = exposed controls, c = unexposed cases, d = unexposed controls

# For controls, exposure should match the general population
control_exposure_prob = exposure_prevalence

# For cases, we need to calculate exposure probability that gives us our target OR
# Using algebra: case_exposure_prob = OR * control_exposure_prob / (1 - control_exposure_prob + OR * control_exposure_prob)
case_exposure_prob = (true_odds_ratio * control_exposure_prob) / (1 - control_exposure_prob + true_odds_ratio * control_exposure_prob)

# Generate exposure status for cases and controls
case_exposure = np.random.binomial(1, case_exposure_prob, n_cases)
control_exposure = np.random.binomial(1, control_exposure_prob, n_controls)

# Create DataFrame for cases and controls
cases = pd.DataFrame({
    'Disease': 1,
    'Exposure': case_exposure
})

controls = pd.DataFrame({
    'Disease': 0,
    'Exposure': control_exposure
})

# Combine cases and controls
case_control_data = pd.concat([cases, controls]).reset_index(drop=True)

# For nicer display, convert binary to categorical
case_control_data['Disease_Status'] = case_control_data['Disease'].map({1: 'Case', 0: 'Control'})
case_control_data['Exposure_Status'] = case_control_data['Exposure'].map({1: 'Exposed', 0: 'Unexposed'})

# Create contingency table
contingency = pd.crosstab(case_control_data['Disease_Status'], case_control_data['Exposure_Status'])
print("Contingency Table:")
print(contingency)

# Calculate cell counts for the 2x2 table
a = sum((case_control_data['Disease'] == 1) & (case_control_data['Exposure'] == 1))  # Exposed cases
b = sum((case_control_data['Disease'] == 0) & (case_control_data['Exposure'] == 1))  # Exposed controls
c = sum((case_control_data['Disease'] == 1) & (case_control_data['Exposure'] == 0))  # Unexposed cases
d = sum((case_control_data['Disease'] == 0) & (case_control_data['Exposure'] == 0))  # Unexposed controls

# Calculate odds ratio
odds_ratio = (a * d) / (b * c)

# Calculate confidence interval for the odds ratio
# Using simple approximation (more advanced methods exist)
log_or = np.log(odds_ratio)
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
ci_lower = np.exp(log_or - 1.96 * se_log_or)
ci_upper = np.exp(log_or + 1.96 * se_log_or)

print("\\nCase-Control Study Results:")
print("-" * 40)
print(f"Cases: {n_cases}, Controls: {n_controls}")
print(f"Exposure in cases: {a} ({a/n_cases:.1%})")
print(f"Exposure in controls: {b} ({b/n_controls:.1%})")
print(f"Calculated Odds Ratio: {odds_ratio:.2f}")
print(f"95% Confidence Interval: ({ci_lower:.2f} - {ci_upper:.2f})")
print(f"True Odds Ratio (set by simulation): {true_odds_ratio:.2f}")

# VISUALIZATIONS

# 1. Bar chart of exposure status by disease status
fig1 = px.histogram(
    case_control_data,
    x='Disease_Status',
    color='Exposure_Status',
    barmode='group',
    title='Exposure Distribution by Disease Status',
    labels={'count': 'Number of People', 'Disease_Status': 'Disease Status'},
    color_discrete_map={'Exposed': 'red', 'Unexposed': 'blue'}
)

fig1.update_layout(height=500)
output_vars['fig1'] = fig1

# 2. Odds ratio visualization
fig2 = go.Figure()

# Add a single bar for the odds ratio
fig2.add_trace(
    go.Bar(
        x=['Odds Ratio'],
        y=[odds_ratio],
        text=[f"{odds_ratio:.2f}"],
        textposition='auto',
        marker_color='green',
        width=0.5
    )
)

# Add error bars for the confidence interval
fig2.add_trace(
    go.Scatter(
        x=['Odds Ratio', 'Odds Ratio'],
        y=[ci_lower, ci_upper],
        mode='markers',
        marker=dict(color='black', size=8),
        showlegend=False
    )
)
fig2.add_shape(
    type="line",
    x0='Odds Ratio', y0=ci_lower,
    x1='Odds Ratio', y1=ci_upper,
    line=dict(color="black", width=2)
)

# Add a reference line at OR = 1 (no association)
fig2.add_shape(
    type="line",
    x0=-0.5, y0=1, x1=0.5, y1=1,
    line=dict(color="red", width=2, dash="dash")
)

# Update layout
fig2.update_layout(
    title='Estimated Odds Ratio with 95% Confidence Interval',
    yaxis_title='Odds Ratio',
    height=500,
    xaxis=dict(range=[-0.5, 0.5])
)

output_vars['fig2'] = fig2
"""
    elif study_design == "Cross-sectional Study":
        initial_code = """# Simulation of a Cross-sectional Study
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
sample_size = 1000           # Number of people in the study
disease_prevalence = 0.15    # Overall disease prevalence (15%)
exposure_prevalence = 0.25   # Overall exposure prevalence (25%)
prevalence_ratio = 2.0       # The ratio of disease prevalence in exposed vs. unexposed

# Set random seed for reproducibility
np.random.seed(42)

# In a cross-sectional study, we measure exposure and outcome at the same time
# We need to generate correlated exposure and disease variables

# First, let's determine what the disease prevalence should be in the exposed and unexposed groups
# Given the overall prevalence and prevalence ratio

# If Pd = overall disease prevalence
# Pe = exposure prevalence
# PR = prevalence ratio
# Pde = disease prevalence in exposed
# Pdu = disease prevalence in unexposed

# We know: Pd = Pe * Pde + (1 - Pe) * Pdu
# And: PR = Pde / Pdu
# Solving for Pdu: Pdu = Pd / (Pe * PR + (1 - Pe))
# Then: Pde = PR * Pdu

prevalence_unexposed = disease_prevalence / (exposure_prevalence * prevalence_ratio + (1 - exposure_prevalence))
prevalence_exposed = prevalence_ratio * prevalence_unexposed

# Ensure probabilities are valid (between 0 and 1)
prevalence_unexposed = min(max(prevalence_unexposed, 0), 1)
prevalence_exposed = min(max(prevalence_exposed, 0), 1)

# Generate exposure status
exposure = np.random.binomial(1, exposure_prevalence, sample_size)

# Generate disease status based on exposure
disease = np.zeros(sample_size, dtype=int)
for i in range(sample_size):
    if exposure[i] == 1:
        disease[i] = np.random.binomial(1, prevalence_exposed)
    else:
        disease[i] = np.random.binomial(1, prevalence_unexposed)

# Create a DataFrame
cross_sectional_data = pd.DataFrame({
    'Exposure': exposure,
    'Disease': disease
})

# Calculate counts for contingency table
a = sum((cross_sectional_data['Exposure'] == 1) & (cross_sectional_data['Disease'] == 1))  # Exposed with disease
b = sum((cross_sectional_data['Exposure'] == 1) & (cross_sectional_data['Disease'] == 0))  # Exposed without disease
c = sum((cross_sectional_data['Exposure'] == 0) & (cross_sectional_data['Disease'] == 1))  # Unexposed with disease
d = sum((cross_sectional_data['Exposure'] == 0) & (cross_sectional_data['Disease'] == 0))  # Unexposed without disease

# Create and display contingency table
contingency = pd.DataFrame({
    'Disease': [a, c],
    'No Disease': [b, d]
}, index=['Exposed', 'Unexposed'])

print("Cross-sectional Study Contingency Table:")
print(contingency)

# Calculate prevalence in each group
prevalence_in_exposed = a / (a + b)
prevalence_in_unexposed = c / (c + d)

# Calculate measures of association
calculated_prevalence_ratio = prevalence_in_exposed / prevalence_in_unexposed
prevalence_difference = prevalence_in_exposed - prevalence_in_unexposed
odds_ratio = (a * d) / (b * c)

print("\\nCross-sectional Study Results:")
print("-" * 40)
print(f"Sample Size: {sample_size}")
print(f"Disease Prevalence (Overall): {(a + c) / sample_size:.1%}")
print(f"Exposure Prevalence (Overall): {(a + b) / sample_size:.1%}")
print(f"Disease Prevalence in Exposed: {prevalence_in_exposed:.1%}")
print(f"Disease Prevalence in Unexposed: {prevalence_in_unexposed:.1%}")
print(f"Prevalence Ratio: {calculated_prevalence_ratio:.2f}")
print(f"Prevalence Difference: {prevalence_difference:.2%}")
print(f"Odds Ratio: {odds_ratio:.2f}")
print(f"True Prevalence Ratio (set by simulation): {prevalence_ratio:.2f}")

# VISUALIZATIONS

# 1. Bar chart of disease prevalence by exposure status
exposure_groups = ['Exposed', 'Unexposed']
prevalence_values = [prevalence_in_exposed, prevalence_in_unexposed]

fig1 = go.Figure(data=[
    go.Bar(
        x=exposure_groups,
        y=prevalence_values,
        text=[f"{p:.1%}" for p in prevalence_values],
        textposition='auto',
        marker_color=['red', 'blue']
    )
])

fig1.update_layout(
    title='Disease Prevalence by Exposure Status',
    yaxis_title='Disease Prevalence',
    xaxis_title='Exposure Status',
    height=500,
    yaxis=dict(tickformat='.0%')
)

output_vars['fig1'] = fig1

# 2. Comparison of measures of association
measures = ['Prevalence Ratio', 'Odds Ratio']
values = [calculated_prevalence_ratio, odds_ratio]

fig2 = go.Figure(data=[
    go.Bar(
        x=measures,
        y=values,
        text=[f"{v:.2f}" for v in values],
        textposition='auto',
        marker_color=['green', 'purple']
    )
])

# Add a reference line at 1 (no association)
fig2.add_shape(
    type="line",
    x0=-0.5, y0=1, x1=1.5, y1=1,
    line=dict(color="black", width=2, dash="dash")
)

fig2.update_layout(
    title='Measures of Association',
    yaxis_title='Value',
    height=500
)

output_vars['fig2'] = fig2

# 3. Mosaic plot (simplified version)
mosaic_data = pd.crosstab(
    cross_sectional_data['Exposure'], 
    cross_sectional_data['Disease'],
    normalize='all'
) * 100  # Convert to percentage

fig3 = go.Figure()

# Define positions for the rectangles
x_positions = [0, mosaic_data.loc[0, 0], 100, 100 - mosaic_data.loc[1, 0]]
y_positions = [0, 0, 0, 0]
widths = [mosaic_data.loc[0, 0] + mosaic_data.loc[0, 1], mosaic_data.loc[1, 0] + mosaic_data.loc[1, 1]]
heights = [mosaic_data.loc[0, 1] / (mosaic_data.loc[0, 0] + mosaic_data.loc[0, 1]) * 100,
           mosaic_data.loc[1, 1] / (mosaic_data.loc[1, 0] + mosaic_data.loc[1, 1]) * 100]

# Add rectangles for "No Disease"
fig3.add_trace(go.Bar(
    x=widths,
    y=[100 - heights[0], 100 - heights[1]],
    marker_color='lightblue',
    name='No Disease',
    hoverinfo='text',
    text=[f"Unexposed, No Disease: {mosaic_data.loc[0, 0]:.1f}%",
          f"Exposed, No Disease: {mosaic_data.loc[1, 0]:.1f}%"],
    orientation='v'
))

# Add rectangles for "Disease"
fig3.add_trace(go.Bar(
    x=widths,
    y=[heights[0], heights[1]],
    marker_color='red',
    name='Disease',
    hoverinfo='text',
    text=[f"Unexposed, Disease: {mosaic_data.loc[0, 1]:.1f}%",
          f"Exposed, Disease: {mosaic_data.loc[1, 1]:.1f}%"],
    orientation='v'
))

fig3.update_layout(
    title='Proportion of Population in Each Category',
    barmode='stack',
    xaxis_title='Exposure',
    yaxis_title='Percentage',
    xaxis=dict(
        tickvals=[widths[0]/2, widths[0] + widths[1]/2],
        ticktext=['Unexposed', 'Exposed']
    ),
    height=500
)

output_vars['fig3'] = fig3
"""
    elif study_design == "Randomized Controlled Trial":
        initial_code = """# Simulation of a Randomized Controlled Trial (RCT)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_participants = 500         # Total number of participants
true_effect = 0.3            # True treatment effect
baseline_risk = 0.2          # Baseline risk of disease in control group
dropout_rate = 0.1           # Proportion of participants who drop out
confounding_factor = False   # Whether to introduce confounding

# Set random seed for reproducibility
np.random.seed(42)

# Generate participant characteristics
age = np.random.normal(50, 15, n_participants)  # Age ~ N(50, 15)
sex = np.random.binomial(1, 0.5, n_participants)  # Sex (0=female, 1=male)

# Assign treatment (randomization)
treatment = np.random.binomial(1, 0.5, n_participants)

# Count participants in each group
n_treatment = sum(treatment)
n_control = n_participants - n_treatment

print(f"Study Participants: {n_participants}")
print(f"Treatment Group: {n_treatment}")
print(f"Control Group: {n_control}")

# Create DataFrame
rct_data = pd.DataFrame({
    'Age': age,
    'Sex': sex,
    'Treatment': treatment
})

# Check if randomization balanced covariates
print("\\nBaseline Characteristics:")
print(rct_data.groupby('Treatment')[['Age', 'Sex']].mean())

# Add outcome (with potential confounding)
if confounding_factor:
    # Outcome depends on treatment AND age
    outcome_prob = baseline_risk + true_effect * treatment + 0.01 * (age - 50)
else:
    # Outcome only depends on treatment
    outcome_prob = baseline_risk + true_effect * treatment

# Ensure probability is between 0 and 1
outcome_prob = np.clip(outcome_prob, 0, 1)

# Generate outcome
outcome = np.random.binomial(1, outcome_prob)
rct_data['Outcome'] = outcome

# Simulate dropout
dropout = np.random.binomial(1, dropout_rate, n_participants)
rct_data['Dropout'] = dropout

# Create "Intention-to-Treat" (ITT) and "Per-Protocol" (PP) datasets
itt_data = rct_data.copy()  # ITT includes everyone
pp_data = rct_data[rct_data['Dropout'] == 0].copy()  # PP only includes those who didn't drop out

# Calculate results for ITT analysis
itt_treatment_outcome = itt_data[itt_data['Treatment'] == 1]['Outcome'].mean()
itt_control_outcome = itt_data[itt_data['Treatment'] == 0]['Outcome'].mean()
itt_effect = itt_treatment_outcome - itt_control_outcome

# Calculate results for PP analysis
pp_treatment_outcome = pp_data[pp_data['Treatment'] == 1]['Outcome'].mean()
pp_control_outcome = pp_data[pp_data['Treatment'] == 0]['Outcome'].mean()
pp_effect = pp_treatment_outcome - pp_control_outcome

print("\\nStudy Results:")
print("-" * 40)
print(f"Dropouts: {sum(dropout)} ({sum(dropout)/n_participants:.1%})")
print(f"True effect (set by simulation): {true_effect:.3f}")

print("\\nIntention-to-Treat Analysis:")
print(f"Outcome in treatment group: {itt_treatment_outcome:.1%}")
print(f"Outcome in control group: {itt_control_outcome:.1%}")
print(f"Treatment effect (risk difference): {itt_effect:.3f}")

print("\\nPer-Protocol Analysis:")
print(f"Outcome in treatment group: {pp_treatment_outcome:.1%}")
print(f"Outcome in control group: {pp_control_outcome:.1%}")
print(f"Treatment effect (risk difference): {pp_effect:.3f}")

# VISUALIZATIONS

# 1. Bar chart of outcome by treatment group (ITT)
treatment_labels = ['Control', 'Treatment']
itt_outcomes = [itt_control_outcome, itt_treatment_outcome]
pp_outcomes = [pp_control_outcome, pp_treatment_outcome]

fig1 = go.Figure(data=[
    go.Bar(
        name='Intention-to-Treat',
        x=treatment_labels,
        y=itt_outcomes,
        text=[f"{o:.1%}" for o in itt_outcomes],
        textposition='auto',
        marker_color='blue'
    ),
    go.Bar(
        name='Per-Protocol',
        x=treatment_labels,
        y=pp_outcomes,
        text=[f"{o:.1%}" for o in pp_outcomes],
        textposition='auto',
        marker_color='green'
    )
])

fig1.update_layout(
    title='Outcome by Treatment Group and Analysis Method',
    yaxis_title='Outcome Rate',
    barmode='group',
    height=500,
    yaxis=dict(tickformat='.0%')
)

output_vars['fig1'] = fig1

# 2. Comparison of effects
effect_methods = ['True Effect', 'ITT Effect', 'PP Effect']
effect_values = [true_effect, itt_effect, pp_effect]

fig2 = go.Figure(data=[
    go.Bar(
        x=effect_methods,
        y=effect_values,
        text=[f"{v:.3f}" for v in effect_values],
        textposition='auto',
        marker_color=['red', 'blue', 'green']
    )
])

# Add a reference line at 0 (no effect)
fig2.add_shape(
    type="line",
    x0=-0.5, y0=0, x1=2.5, y1=0,
    line=dict(color="black", width=2, dash="dash")
)

fig2.update_layout(
    title='Comparison of Effect Estimates',
    yaxis_title='Effect Size (Risk Difference)',
    height=500
)

output_vars['fig2'] = fig2

# 3. CONSORT flow diagram (simplified)
consort_labels = ['Randomized', 'Treatment Group', 'Control Group', 
                 'Completed (Treatment)', 'Completed (Control)']

treatment_dropout = sum((rct_data['Treatment'] == 1) & (rct_data['Dropout'] == 1))
control_dropout = sum((rct_data['Treatment'] == 0) & (rct_data['Dropout'] == 1))

consort_values = [n_participants, n_treatment, n_control, 
                  n_treatment - treatment_dropout, n_control - control_dropout]

fig3 = go.Figure(data=[
    go.Bar(
        x=consort_labels,
        y=consort_values,
        text=[f"{v}" for v in consort_values],
        textposition='auto',
        marker_color=['purple', 'blue', 'orange', 'green', 'red']
    )
])

fig3.update_layout(
    title='CONSORT Flow Diagram',
    yaxis_title='Number of Participants',
    height=500
)

output_vars['fig3'] = fig3
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
            if study_design == "Cohort Study":
                st.markdown("""
                **Try modifying:**
                - Change `relative_risk` to see how it affects disease occurrence
                - Adjust `followup_years` to see how study duration affects results
                - Modify `population_size` to see the effect of sample size
                - Change `baseline_risk` to simulate different disease frequencies
                
                **Simple Challenges:**
                1. Create a protective exposure (RR < 1) by setting `relative_risk` to 0.5
                2. Simulate a rare disease by lowering `baseline_risk` to 0.01
                3. See how increasing the follow-up time affects the results
                4. Find how large a cohort you need to detect a small effect (RR = 1.2)
                """)
            elif study_design == "Case-Control Study":
                st.markdown("""
                **Try modifying:**
                - Change `true_odds_ratio` to different values
                - Adjust the ratio of controls to cases by changing `n_controls`
                - Modify `exposure_prevalence` to simulate different exposures
                - Try a small number of cases to see how it affects precision
                
                **Simple Challenges:**
                1. Create a situation where the exposure is protective (OR < 1)
                2. Find how many controls you need for a narrow confidence interval
                3. See what happens when the exposure is very rare
                4. Compare case-control with different control:case ratios
                """)
            elif study_design == "Cross-sectional Study":
                st.markdown("""
                **Try modifying:**
                - Change `prevalence_ratio` to see how it affects the measures of association
                - Adjust `disease_prevalence` from rare to common
                - Modify `exposure_prevalence` to simulate different exposure scenarios
                - Change `sample_size` to see how it affects the stability of results
                
                **Simple Challenges:**
                1. Make the disease very rare (prevalence < 5%) and see how PR and OR compare
                2. Make the disease very common (prevalence > 30%) and compare PR and OR
                3. Create a protective association (PR < 1)
                4. Find what sample size you need for stable results
                """)
            elif study_design == "Randomized Controlled Trial":
                st.markdown("""
                **Try modifying:**
                - Change `true_effect` to simulate different treatment effects
                - Adjust `dropout_rate` to see how it affects ITT vs PP analysis
                - Toggle `confounding_factor` to see how randomization controls for confounding
                - Change `baseline_risk` to simulate different disease frequencies
                
                **Simple Challenges:**
                1. Create a harmful treatment (negative effect)
                2. See how high dropout rates affect the difference between ITT and PP
                3. Turn on confounding and see if randomization balanced the groups
                4. Simulate a very small effect and see if you can detect it
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
            for i in range(1, 4):  # Check for fig1, fig2, fig3
                fig_key = f'fig{i}'
                if fig_key in output_vars:
                    st.plotly_chart(output_vars[fig_key], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section - simplified for beginners
    st.subheader("Key Concepts")
    
    if study_design == "Cohort Study":
        st.markdown("""
        ### Understanding Cohort Studies:
        
        1. **Basic Design:**
           - Starts with exposure status (exposed vs. unexposed groups)
           - Follows participants over time to see who develops the outcome
           - Measures disease incidence in each group
        
        2. **Key Measures:**
           - **Risk**: Probability of developing disease during follow-up
           - **Risk Ratio (RR)**: Ratio of risk in exposed to risk in unexposed
           - **Risk Difference (RD)**: Absolute difference in risks
           - **Incidence Rate**: Cases per person-time of follow-up
        
        3. **Strengths:**
           - Can establish temporal sequence (exposure before outcome)
           - Can study multiple outcomes for the same exposure
           - Can directly measure incidence and risk
           - Good for common exposures
        
        4. **Limitations:**
           - Inefficient for rare diseases
           - Can be time-consuming and expensive
           - Loss to follow-up may introduce bias
           - May require large sample sizes
        """)
    elif study_design == "Case-Control Study":
        st.markdown("""
        ### Understanding Case-Control Studies:
        
        1. **Basic Design:**
           - Starts with outcome status (cases vs. controls)
           - Looks backward to measure past exposures
           - Compares exposure history between groups
        
        2. **Key Measures:**
           - **Odds Ratio (OR)**: Ratio of odds of exposure in cases to odds in controls
           - When disease is rare, OR approximates Risk Ratio
           - Cannot directly calculate risk or incidence
        
        3. **Strengths:**
           - Efficient for rare diseases
           - Relatively quick and inexpensive
           - Can study multiple exposures
           - Requires fewer participants than cohort studies
        
        4. **Limitations:**
           - Prone to selection bias
           - Cannot directly measure incidence
           - Recall bias in exposure assessment
           - Difficult to establish temporal sequence
        """)
    elif study_design == "Cross-sectional Study":
        st.markdown("""
        ### Understanding Cross-sectional Studies:
        
        1. **Basic Design:**
           - Measures exposure and outcome at the same time
           - Snapshot of a population at one point in time
           - Studies the prevalence of disease and exposures
        
        2. **Key Measures:**
           - **Prevalence**: Proportion of population with disease at a given time
           - **Prevalence Ratio (PR)**: Ratio of disease prevalence in exposed vs. unexposed
           - **Prevalence Odds Ratio (POR)**: Ratio of odds of disease in exposed vs. unexposed
           - **Prevalence Difference**: Absolute difference in disease prevalence
        
        3. **Strengths:**
           - Quick and relatively inexpensive
           - No loss to follow-up
           - Good for measuring burden of disease
           - Can study multiple exposures and outcomes
        
        4. **Limitations:**
           - Cannot establish temporal sequence (which came first)
           - Measures prevalence, not incidence
           - Subject to prevalence-incidence bias
           - Not suitable for rare diseases
        """)
    elif study_design == "Randomized Controlled Trial":
        st.markdown("""
        ### Understanding Randomized Controlled Trials:
        
        1. **Basic Design:**
           - Participants randomly assigned to treatment or control
           - Follows participants to measure outcomes
           - Intervention is controlled by the researcher
        
        2. **Key Measures:**
           - **Absolute Risk Reduction/Increase**: Difference in outcome rates
           - **Relative Risk**: Ratio of outcome rates
           - **Number Needed to Treat**: 1/Absolute Risk Reduction
        
        3. **Key Analysis Types:**
           - **Intention-to-Treat (ITT)**: Analyzes everyone as randomized
           - **Per-Protocol (PP)**: Analyzes only those who completed the protocol
        
        4. **Strengths:**
           - Minimizes confounding through randomization
           - Can establish causality
           - Reduces selection bias
           - Gold standard for testing interventions
        
        5. **Limitations:**
           - Can be expensive and time-consuming
           - Ethical constraints on what can be studied
           - May have limited generalizability
           - Dropout may introduce bias
        """)

def comparing_study_designs_lesson():
    st.header("Comparing Study Designs")
    
    st.markdown("""
    ### How Different Study Designs Compare
    
    Different epidemiological study designs have unique strengths, limitations, and uses.
    This exercise explores the same research question using different study designs.
    """)
    
    # Initial code example - simplified for beginners
    initial_code = """# Comparing different epidemiological study designs
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
population_size = 100000     # Size of the source population
disease_incidence = 0.05     # Annual incidence of disease (5%)
true_relative_risk = 2.5     # True effect of exposure on disease
exposure_prevalence = 0.3    # Prevalence of exposure in the population

# Set random seed for reproducibility
np.random.seed(42)

# First, let's simulate a source population for all our study designs
# We'll create a population with an exposure that influences disease risk

# 1. Generate exposure status
exposure = np.random.binomial(1, exposure_prevalence, population_size)

# 2. Generate disease status based on exposure
# We'll simulate a two-year time frame
baseline_risk = disease_incidence
exposed_risk = baseline_risk * true_relative_risk

# Ensure risks aren't above 1
exposed_risk = min(exposed_risk, 0.99)
unexposed_risk = min(baseline_risk, 0.99)

# Generate time to disease
time_to_disease = np.zeros(population_size)
for i in range(population_size):
    if exposure[i] == 1:
        # Exposed individuals: higher risk
        time_to_disease[i] = np.random.exponential(1/exposed_risk)
    else:
        # Unexposed individuals: baseline risk
        time_to_disease[i] = np.random.exponential(1/unexposed_risk)

# Determine disease status at 2 years
disease = time_to_disease <= 2
time_observed = np.minimum(time_to_disease, 2)

# Create the full population dataset
population_data = pd.DataFrame({
    'ID': range(population_size),
    'Exposure': exposure,
    'Disease': disease,
    'Time_to_Disease': time_to_disease,
    'Time_Observed': time_observed
})

# Print population statistics
n_exposed = sum(exposure)
n_unexposed = population_size - n_exposed
n_disease = sum(disease)
n_exposed_disease = sum(exposure & disease)
n_unexposed_disease = sum((~exposure) & disease)

print("SOURCE POPULATION:")
print(f"Population Size: {population_size}")
print(f"Exposed: {n_exposed} ({n_exposed/population_size:.1%})")
print(f"Unexposed: {n_unexposed} ({n_unexposed/population_size:.1%})")
print(f"Disease Cases: {n_disease} ({n_disease/population_size:.1%})")
print(f"Exposed with Disease: {n_exposed_disease} ({n_exposed_disease/n_exposed:.1%})")
print(f"Unexposed with Disease: {n_unexposed_disease} ({n_unexposed_disease/n_unexposed:.1%})")

print("\\nTRUE MEASURES OF ASSOCIATION IN POPULATION:")
true_risk_exposed = n_exposed_disease / n_exposed
true_risk_unexposed = n_unexposed_disease / n_unexposed
true_risk_ratio = true_risk_exposed / true_risk_unexposed
true_risk_difference = true_risk_exposed - true_risk_unexposed
true_odds_exposed = n_exposed_disease / (n_exposed - n_exposed_disease)
true_odds_unexposed = n_unexposed_disease / (n_unexposed - n_unexposed_disease)
true_odds_ratio = true_odds_exposed / true_odds_unexposed

print(f"Risk in Exposed: {true_risk_exposed:.1%}")
print(f"Risk in Unexposed: {true_risk_unexposed:.1%}")
print(f"Risk Ratio: {true_risk_ratio:.2f}")
print(f"Risk Difference: {true_risk_difference:.1%}")
print(f"Odds Ratio: {true_odds_ratio:.2f}")

print("\\nNOW LET'S SIMULATE DIFFERENT STUDY DESIGNS")
print("-" * 50)

# 1. COHORT STUDY
# Sample a cohort from the population (5% of population)
cohort_size = int(population_size * 0.05)
cohort_indices = np.random.choice(population_size, cohort_size, replace=False)
cohort_data = population_data.iloc[cohort_indices].copy()

# Calculate cohort study results
cohort_exposed = sum(cohort_data['Exposure'])
cohort_unexposed = cohort_size - cohort_exposed
cohort_exposed_disease = sum((cohort_data['Exposure'] == 1) & (cohort_data['Disease'] == 1))
cohort_unexposed_disease = sum((cohort_data['Exposure'] == 0) & (cohort_data['Disease'] == 1))

cohort_risk_exposed = cohort_exposed_disease / cohort_exposed
cohort_risk_unexposed = cohort_unexposed_disease / cohort_unexposed
cohort_risk_ratio = cohort_risk_exposed / cohort_risk_unexposed
cohort_risk_difference = cohort_risk_exposed - cohort_risk_unexposed

print("COHORT STUDY RESULTS:")
print(f"Cohort Size: {cohort_size}")
print(f"Exposed with Disease: {cohort_exposed_disease} / {cohort_exposed} = {cohort_risk_exposed:.1%}")
print(f"Unexposed with Disease: {cohort_unexposed_disease} / {cohort_unexposed} = {cohort_risk_unexposed:.1%}")
print(f"Risk Ratio: {cohort_risk_ratio:.2f}")
print(f"Risk Difference: {cohort_risk_difference:.1%}")

# 2. CASE-CONTROL STUDY
# Sample all cases and a sample of controls
n_cases_sampled = min(1000, n_disease)  # Cap at 1000 cases
n_controls_sampled = n_cases_sampled * 2  # 2 controls per case

# Sample cases
case_indices = np.random.choice(
    population_data[population_data['Disease'] == 1].index,
    n_cases_sampled,
    replace=False
)

# Sample controls
control_indices = np.random.choice(
    population_data[population_data['Disease'] == 0].index,
    n_controls_sampled,
    replace=False
)

# Combine cases and controls
case_control_indices = np.concatenate([case_indices, control_indices])
case_control_data = population_data.iloc[case_control_indices].copy()

# Calculate case-control study results
cc_cases_exposed = sum((case_control_data['Disease'] == 1) & (case_control_data['Exposure'] == 1))
cc_cases_unexposed = n_cases_sampled - cc_cases_exposed
cc_controls_exposed = sum((case_control_data['Disease'] == 0) & (case_control_data['Exposure'] == 1))
cc_controls_unexposed = n_controls_sampled - cc_controls_exposed

cc_odds_exposed = cc_cases_exposed / cc_controls_exposed
cc_odds_unexposed = cc_cases_unexposed / cc_controls_unexposed
cc_odds_ratio = (cc_cases_exposed * cc_controls_unexposed) / (cc_cases_unexposed * cc_controls_exposed)

print("\\nCASE-CONTROL STUDY RESULTS:")
print(f"Cases: {n_cases_sampled}, Controls: {n_controls_sampled}")
print(f"Exposure in Cases: {cc_cases_exposed} / {n_cases_sampled} = {cc_cases_exposed/n_cases_sampled:.1%}")
print(f"Exposure in Controls: {cc_controls_exposed} / {n_controls_sampled} = {cc_controls_exposed/n_controls_sampled:.1%}")
print(f"Odds Ratio: {cc_odds_ratio:.2f}")

# 3. CROSS-SECTIONAL STUDY
# Take a snapshot of the population
cross_sectional_size = int(population_size * 0.05)
cs_indices = np.random.choice(population_size, cross_sectional_size, replace=False)
cs_data = population_data.iloc[cs_indices].copy()

# Calculate cross-sectional study results
cs_exposed = sum(cs_data['Exposure'])
cs_unexposed = cross_sectional_size - cs_exposed
cs_exposed_disease = sum((cs_data['Exposure'] == 1) & (cs_data['Disease'] == 1))
cs_unexposed_disease = sum((cs_data['Exposure'] == 0) & (cs_data['Disease'] == 1))

cs_prevalence_exposed = cs_exposed_disease / cs_exposed
cs_prevalence_unexposed = cs_unexposed_disease / cs_unexposed
cs_prevalence_ratio = cs_prevalence_exposed / cs_prevalence_unexposed
cs_prevalence_difference = cs_prevalence_exposed - cs_prevalence_unexposed
cs_odds_ratio = (cs_exposed_disease * (cs_unexposed - cs_unexposed_disease)) / ((cs_exposed - cs_exposed_disease) * cs_unexposed_disease)

print("\\nCROSS-SECTIONAL STUDY RESULTS:")
print(f"Sample Size: {cross_sectional_size}")
print(f"Disease in Exposed: {cs_exposed_disease} / {cs_exposed} = {cs_prevalence_exposed:.1%}")
print(f"Disease in Unexposed: {cs_unexposed_disease} / {cs_unexposed} = {cs_prevalence_unexposed:.1%}")
print(f"Prevalence Ratio: {cs_prevalence_ratio:.2f}")
print(f"Prevalence Difference: {cs_prevalence_difference:.1%}")
print(f"Odds Ratio: {cs_odds_ratio:.2f}")

# 4. RANDOMIZED TRIAL (SIMPLIFIED)
# Simulate a randomized trial with perfect compliance
trial_size = 2000
treatment_indices = np.random.choice(population_size, trial_size, replace=False)
trial_data = population_data.iloc[treatment_indices].copy()

# Randomly assign treatment (ignore original exposure)
trial_data['Treatment'] = np.random.binomial(1, 0.5, trial_size)

# Assume treatment has the same effect as the exposure in the population
# Generate new outcome data based on treatment
baseline_risk_trial = disease_incidence * 2  # For 2 years
treatment_risk_trial = baseline_risk_trial * true_relative_risk

# Generate time to disease based on treatment
time_to_disease_trial = np.zeros(trial_size)
for i in range(trial_size):
    if trial_data['Treatment'].iloc[i] == 1:
        # Treated individuals: higher or lower risk depending on RR
        time_to_disease_trial[i] = np.random.exponential(1/treatment_risk_trial)
    else:
        # Control individuals: baseline risk
        time_to_disease_trial[i] = np.random.exponential(1/baseline_risk_trial)

# Determine disease status at 2 years
trial_data['Disease_Trial'] = time_to_disease_trial <= 2

# Calculate trial results
trial_treated = sum(trial_data['Treatment'])
trial_control = trial_size - trial_treated
trial_treated_disease = sum((trial_data['Treatment'] == 1) & (trial_data['Disease_Trial'] == 1))
trial_control_disease = sum((trial_data['Treatment'] == 0) & (trial_data['Disease_Trial'] == 1))

trial_risk_treated = trial_treated_disease / trial_treated
trial_risk_control = trial_control_disease / trial_control
trial_risk_ratio = trial_risk_treated / trial_risk_control
trial_risk_difference = trial_risk_treated - trial_risk_control

print("\\nRANDOMIZED TRIAL RESULTS:")
print(f"Trial Size: {trial_size}")
print(f"Disease in Treated: {trial_treated_disease} / {trial_treated} = {trial_risk_treated:.1%}")
print(f"Disease in Control: {trial_control_disease} / {trial_control} = {trial_risk_control:.1%}")
print(f"Risk Ratio: {trial_risk_ratio:.2f}")
print(f"Risk Difference: {trial_risk_difference:.1%}")

# VISUALIZATIONS

# 1. Comparison of key measures across study designs
study_designs = ['True Population', 'Cohort Study', 'Case-Control', 'Cross-sectional', 'Randomized Trial']
risk_ratios = [true_risk_ratio, cohort_risk_ratio, cc_odds_ratio, cs_prevalence_ratio, trial_risk_ratio]
measure_labels = [
    f"RR: {true_risk_ratio:.2f}",
    f"RR: {cohort_risk_ratio:.2f}",
    f"OR: {cc_odds_ratio:.2f}",
    f"PR: {cs_prevalence_ratio:.2f}",
    f"RR: {trial_risk_ratio:.2f}"
]

fig1 = go.Figure(data=[
    go.Bar(
        x=study_designs,
        y=risk_ratios,
        text=measure_labels,
        textposition='auto',
        marker_color=['purple', 'blue', 'green', 'orange', 'red']
    )
])

# Add a reference line at 1 (no association)
fig1.add_shape(
    type="line",
    x0=-0.5, y0=1, x1=4.5, y1=1,
    line=dict(color="black", width=2, dash="dash")
)

fig1.update_layout(
    title='Comparison of Measures Across Study Designs',
    yaxis_title='Value',
    height=500,
    yaxis=dict(range=[0, max(risk_ratios) * 1.1])
)

output_vars['fig1'] = fig1

# 2. Sample size comparison
study_sizes = [population_size, cohort_size, n_cases_sampled + n_controls_sampled, cross_sectional_size, trial_size]
size_labels = [
    f"{population_size:,}",
    f"{cohort_size:,}",
    f"{n_cases_sampled + n_controls_sampled:,}",
    f"{cross_sectional_size:,}",
    f"{trial_size:,}"
]

fig2 = go.Figure(data=[
    go.Bar(
        x=study_designs,
        y=study_sizes,
        text=size_labels,
        textposition='auto',
        marker_color=['purple', 'blue', 'green', 'orange', 'red']
    )
])

fig2.update_layout(
    title='Sample Size by Study Design',
    yaxis_title='Number of Participants',
    height=500,
    yaxis=dict(type='log')  # Log scale due to large differences
)

output_vars['fig2'] = fig2

# 3. Comparison of exposure and outcome relationship across designs

# Create a DataFrame for comparison
comparison_data = []

# Add population data
population_exposed_pct = n_exposed / population_size * 100
population_disease_pct = n_disease / population_size * 100
comparison_data.append({
    'Study Design': 'Population',
    'Exposed (%)': population_exposed_pct,
    'Disease (%)': population_disease_pct,
    'Method': 'True Values'
})

# Add cohort data
cohort_exposed_pct = cohort_exposed / cohort_size * 100
cohort_disease_pct = sum(cohort_data['Disease']) / cohort_size * 100
comparison_data.append({
    'Study Design': 'Cohort',
    'Exposed (%)': cohort_exposed_pct,
    'Disease (%)': cohort_disease_pct,
    'Method': 'Longitudinal'
})

# Add case-control data (special consideration since we deliberately sample by disease status)
cc_exposed_pct = (cc_cases_exposed + cc_controls_exposed) / (n_cases_sampled + n_controls_sampled) * 100
# By design, disease prevalence in case-control is not representative
cc_disease_pct = n_cases_sampled / (n_cases_sampled + n_controls_sampled) * 100
comparison_data.append({
    'Study Design': 'Case-Control',
    'Exposed (%)': cc_exposed_pct,
    'Disease (%)': cc_disease_pct,
    'Method': 'Retrospective'
})

# Add cross-sectional data
cs_exposed_pct = cs_exposed / cross_sectional_size * 100
cs_disease_pct = (cs_exposed_disease + cs_unexposed_disease) / cross_sectional_size * 100
comparison_data.append({
    'Study Design': 'Cross-sectional',
    'Exposed (%)': cs_exposed_pct,
    'Disease (%)': cs_disease_pct,
    'Method': 'Single Timepoint'
})

# Add trial data (exposure is randomized treatment)
trial_exposed_pct = trial_treated / trial_size * 100
trial_disease_pct = sum(trial_data['Disease_Trial']) / trial_size * 100
comparison_data.append({
    'Study Design': 'Trial',
    'Exposed (%)': trial_exposed_pct,
    'Disease (%)': trial_disease_pct,
    'Method': 'Experimental'
})

comparison_df = pd.DataFrame(comparison_data)

# Create a grouped bar chart
fig3 = go.Figure()

fig3.add_trace(go.Bar(
    x=comparison_df['Study Design'],
    y=comparison_df['Exposed (%)'],
    name='Exposed (%)',
    marker_color='blue',
    text=[f"{x:.1f}%" for x in comparison_df['Exposed (%)']],
    textposition='auto'
))

fig3.add_trace(go.Bar(
    x=comparison_df['Study Design'],
    y=comparison_df['Disease (%)'],
    name='Disease (%)',
    marker_color='red',
    text=[f"{x:.1f}%" for x in comparison_df['Disease (%)']],
    textposition='auto'
))

fig3.update_layout(
    title='Exposure and Disease Prevalence by Study Design',
    yaxis_title='Percentage (%)',
    height=500,
    barmode='group'
)

output_vars['fig3'] = fig3
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
            - Change `true_relative_risk` to see how different effect sizes appear in different designs
            - Adjust `disease_incidence` to see how disease frequency affects study results
            - Modify `exposure_prevalence` to simulate different exposures
            - Change sample sizes for different study designs
            
            **Simple Challenges:**
            1. Make the disease very rare (incidence < 1%) and see which study design performs best
            2. Create a protective exposure (RR < 1) and see how it appears in each design
            3. Compare the precision of results from different study designs
            4. Find the minimum sample size needed for each design to detect the effect
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
            for i in range(1, 4):  # Check for fig1, fig2, fig3
                fig_key = f'fig{i}'
                if fig_key in output_vars:
                    st.plotly_chart(output_vars[fig_key], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section - simplified for beginners
    st.subheader("Key Concepts")
    st.markdown("""
    ### Comparing Study Designs:
    
    1. **Strength of Evidence:**
       - **Randomized Trials**: Strongest evidence (experimental, controls for confounding)
       - **Cohort Studies**: Strong observational evidence (establishes temporality)
       - **Case-Control Studies**: Moderate evidence (efficient but prone to biases)
       - **Cross-sectional Studies**: Weaker evidence (cannot establish temporality)
    
    2. **Efficiency for Different Scenarios:**
       - **Rare Diseases**: Case-control studies are most efficient
       - **Rare Exposures**: Cohort studies are most efficient
       - **Multiple Outcomes**: Cohort studies can examine many outcomes
       - **Multiple Exposures**: Case-control and cross-sectional can examine many exposures
       - **Immediate Results Needed**: Cross-sectional studies are quickest
    
    3. **Measures of Association:**
       - **Cohort & RCT**: Can directly calculate Risk Ratio (RR) and Risk Difference (RD)
       - **Case-Control**: Can only calculate Odds Ratio (OR)
       - **Cross-sectional**: Calculates Prevalence Ratio (PR) and Prevalence Odds Ratio (POR)
       - When disease is rare, OR approximates RR
    
    4. **Practical Considerations:**
       - **Cost**: Generally RCT > Cohort > Case-Control > Cross-sectional
       - **Time**: Generally RCT > Cohort > Case-Control > Cross-sectional
       - **Sample Size**: Generally Cohort > RCT > Cross-sectional > Case-Control
       - **Ethical Constraints**: RCTs have the most ethical limitations
    
    5. **Bias Concerns:**
       - **Selection Bias**: Most problematic in case-control studies
       - **Loss to Follow-up**: Issue in cohort studies and RCTs
       - **Recall Bias**: Major concern in case-control studies
       - **Confounding**: Concern in all observational designs, controlled by randomization in RCTs
    """)

if __name__ == "__main__":
    app()