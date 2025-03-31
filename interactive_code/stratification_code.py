import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
from contextlib import redirect_stdout
import sys
import traceback

def app():
    st.title("Stratification Analysis: Interactive Code Laboratory")
    
    st.markdown("""
    ## Learn by Coding: Stratification in Clinical Research
    
    This interactive code laboratory allows you to explore how stratification works through simple
    Python examples. Stratification is a key technique in epidemiology and clinical research for 
    understanding how relationships vary across different groups.
    
    Choose an example below to get started:
    """)
    
    # Example selection
    example = st.selectbox(
        "Select an example:",
        ["Drug Response by Age Group", 
         "Side Effect Rates by Gender",
         "Treatment Efficacy by Disease Severity"]
    )
    
    # Display the selected example
    if example == "Drug Response by Age Group":
        drug_response_example()
    elif example == "Side Effect Rates by Gender":
        side_effect_example()
    elif example == "Treatment Efficacy by Disease Severity":
        disease_severity_example()

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
                'plt': plt,
                'px': px,
                'go': go,
                'stats': stats,
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

def drug_response_example():
    st.header("Drug Response by Age Group")
    
    st.markdown("""
    ### Understanding Drug Response Across Different Age Groups
    
    Many medications work differently depending on a patient's age. This example explores
    how we can use stratification to examine the relationship between drug dosage and blood
    pressure reduction in different age groups.
    
    Let's simulate some data and analyze it:
    """)
    
    # Initial code example
    initial_code = """# Analyzing drug response data stratified by age group
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(42)

# MODIFY THESE PARAMETERS TO SEE DIFFERENT EFFECTS
young_effect = 1.2    # Effect of drug in young patients (mmHg reduction per mg)
middle_effect = 0.8   # Effect of drug in middle-aged patients
elderly_effect = 0.4  # Effect of drug in elderly patients
sample_size = 30      # Number of patients per age group

# Function to generate synthetic patient data
def generate_patient_data(effect_size, n_patients, age_group):
    # Generate dosages between 10 and 50 mg
    dosage = np.random.uniform(10, 50, n_patients)
    
    # Generate blood pressure reduction based on dosage and effect size
    # Add some random noise to simulate individual variation
    bp_reduction = effect_size * dosage + np.random.normal(0, 5, n_patients)
    
    # Create dataframe
    return pd.DataFrame({
        'Age Group': [age_group] * n_patients,
        'Dosage (mg)': dosage,
        'BP Reduction (mmHg)': bp_reduction
    })

# Generate data for each age group
young_data = generate_patient_data(young_effect, sample_size, 'Young (18-40)')
middle_data = generate_patient_data(middle_effect, sample_size, 'Middle (41-65)')
elderly_data = generate_patient_data(elderly_effect, sample_size, 'Elderly (66+)')

# Combine all data
all_data = pd.concat([young_data, middle_data, elderly_data], ignore_index=True)

# Print summary statistics
print("Data Summary (First 5 rows):")
print(all_data.head())

print("\\nSummary statistics by age group:")
print(all_data.groupby('Age Group')['BP Reduction (mmHg)'].describe())

# Analyze the unstratified data (overall relationship)
# Calculate correlation between dosage and BP reduction
unstratified_corr, p_value = stats.pearsonr(
    all_data['Dosage (mg)'], 
    all_data['BP Reduction (mmHg)']
)

print(f"\\nUnstratified analysis:")
print(f"Correlation: {unstratified_corr:.3f} (p-value: {p_value:.3f})")

# Create unstratified scatter plot
fig1 = px.scatter(
    all_data, 
    x='Dosage (mg)', 
    y='BP Reduction (mmHg)', 
    title='Overall Relationship: Dosage vs. BP Reduction',
    trendline='ols'  # Add ordinary least squares regression line
)

# Customize the plot
fig1.update_layout(
    xaxis_title='Dosage (mg)',
    yaxis_title='Blood Pressure Reduction (mmHg)'
)

# Store the figure for display
output_vars['fig1'] = fig1

# Now analyze the stratified data
# Calculate correlation for each age group
stratified_corr = {}
for age_group in all_data['Age Group'].unique():
    group_data = all_data[all_data['Age Group'] == age_group]
    corr, p_val = stats.pearsonr(
        group_data['Dosage (mg)'], 
        group_data['BP Reduction (mmHg)']
    )
    stratified_corr[age_group] = (corr, p_val)
    print(f"\\n{age_group} group:")
    print(f"Correlation: {corr:.3f} (p-value: {p_val:.3f})")

# Create stratified scatter plot
fig2 = px.scatter(
    all_data, 
    x='Dosage (mg)', 
    y='BP Reduction (mmHg)', 
    color='Age Group',
    title='Stratified Analysis: Dosage vs. BP Reduction by Age Group',
    trendline='ols'  # Add trend lines
)

# Customize the stratified plot
fig2.update_layout(
    xaxis_title='Dosage (mg)',
    yaxis_title='Blood Pressure Reduction (mmHg)'
)

# Store the figure for display
output_vars['fig2'] = fig2

# Create a faceted plot (separate plot for each group)
fig3 = px.scatter(
    all_data, 
    x='Dosage (mg)', 
    y='BP Reduction (mmHg)', 
    color='Age Group',
    facet_col='Age Group',
    title='Faceted View: Dosage vs. BP Reduction by Age Group',
    trendline='ols'
)

# Store the faceted figure
output_vars['fig3'] = fig3

# Calculate mean BP reduction for each age group
mean_by_group = all_data.groupby('Age Group')['BP Reduction (mmHg)'].mean().reset_index()

# Create a bar chart of average BP reduction by age group
fig4 = px.bar(
    mean_by_group,
    x='Age Group',
    y='BP Reduction (mmHg)',
    title='Average BP Reduction by Age Group',
    color='Age Group'
)

# Store the bar chart
output_vars['fig4'] = fig4
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see the results:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Try yourself"):
            st.markdown("""
            **Try modifying:**
            - Change the `young_effect`, `middle_effect`, and `elderly_effect` values to create different patterns
            - Try setting all effects to be equal and see what happens
            - Increase `sample_size` to see how it affects the clarity of the relationships
            
            **Challenge yourself:**
            1. What happens if you make the effect negative for one age group?
            2. Try adding more random noise to see how it affects the analysis
            3. Can you modify the code to add a fourth age group?
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
            if 'fig1' in output_vars:
                st.plotly_chart(output_vars['fig1'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
                
            if 'fig3' in output_vars:
                st.plotly_chart(output_vars['fig3'], use_container_width=True)
                
            if 'fig4' in output_vars:
                st.plotly_chart(output_vars['fig4'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Insights")
    st.markdown("""
    ### What This Analysis Shows:
    
    1. **Overall vs. Stratified Analysis**: 
       - The overall relationship (first plot) gives the average effect across all age groups
       - The stratified analysis (second and third plots) reveals how the drug effect differs by age
    
    2. **Effect Modification**: 
       - When the relationship between dose and response differs across strata, we call this "effect modification"
       - In this example, younger patients show a stronger response to the drug than elderly patients
    
    3. **Clinical Implications**:
       - Personalized dosing: Younger patients may need lower doses to achieve the same effect
       - Efficacy concerns: If the drug effect is too weak in elderly patients, alternative therapies might be needed
       - Safety monitoring: Different age groups may require different monitoring approaches
    
    ### Why Stratification Matters:
    
    Without stratification, we would only see the average effect and might miss important differences between age groups. These differences could affect:
    
    - Clinical decision-making
    - Drug dosing guidelines
    - Benefit-risk assessments
    - Treatment recommendations
    
    Always consider stratifying by important patient characteristics when analyzing drug effects!
    """)

def side_effect_example():
    st.header("Side Effect Rates by Gender")
    
    st.markdown("""
    ### Analyzing Side Effect Rates Stratified by Gender
    
    Men and women often respond differently to medications. This example explores how to analyze
    the relationship between drug dosage and the rate of side effects, stratified by gender.
    
    Let's explore this through code:
    """)
    
    # Initial code example
    initial_code = """# Analyzing side effect rates stratified by gender
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(123)

# MODIFY THESE PARAMETERS TO SEE DIFFERENT PATTERNS
male_base_rate = 0.05     # Base rate of side effects in males (at lowest dose)
female_base_rate = 0.10   # Base rate of side effects in females (at lowest dose)
male_dose_effect = 0.02   # Increase in side effect probability per 10mg dose for males
female_dose_effect = 0.03 # Increase in side effect probability per 10mg dose for females
n_patients = 200          # Total patients (split between males and females)

# Generate patient data
def generate_side_effect_data(base_rate, dose_effect, n_patients, gender):
    # Create different dosage groups (10, 20, 30, 40, 50 mg)
    dosage_groups = np.array([10, 20, 30, 40, 50])
    # Number of patients per dosage group (roughly equal)
    patients_per_group = n_patients // len(dosage_groups)
    
    data = []
    for dose in dosage_groups:
        # Calculate side effect probability for this dose
        side_effect_prob = base_rate + dose_effect * (dose // 10)
        # Cap probability at 1.0 (100%)
        side_effect_prob = min(side_effect_prob, 1.0)
        
        # Generate side effects (1 = had side effect, 0 = no side effect)
        for i in range(patients_per_group):
            had_side_effect = np.random.random() < side_effect_prob
            data.append({
                'Gender': gender,
                'Dosage (mg)': dose,
                'Had Side Effect': 1 if had_side_effect else 0
            })
    
    return pd.DataFrame(data)

# Generate data for males and females
male_data = generate_side_effect_data(male_base_rate, male_dose_effect, n_patients // 2, 'Male')
female_data = generate_side_effect_data(female_base_rate, female_dose_effect, n_patients // 2, 'Female')

# Combine data
all_data = pd.concat([male_data, female_data], ignore_index=True)

# Print data summary
print("Data summary (first 5 rows):")
print(all_data.head())

# Calculate side effect rates for the overall population by dosage
overall_rates = all_data.groupby('Dosage (mg)')['Had Side Effect'].mean().reset_index()
overall_rates['Side Effect Rate (%)'] = overall_rates['Had Side Effect'] * 100

print("\\nOverall side effect rates by dosage:")
print(overall_rates)

# Calculate side effect rates stratified by gender
stratified_rates = all_data.groupby(['Gender', 'Dosage (mg)'])['Had Side Effect'].mean().reset_index()
stratified_rates['Side Effect Rate (%)'] = stratified_rates['Had Side Effect'] * 100

print("\\nSide effect rates stratified by gender:")
print(stratified_rates)

# Create a bar chart of overall side effect rates
fig1 = px.bar(
    overall_rates,
    x='Dosage (mg)',
    y='Side Effect Rate (%)',
    title='Overall Side Effect Rates by Dosage',
    text_auto='.1f'  # Show percentages on bars
)

# Update layout
fig1.update_layout(
    xaxis_title='Dosage (mg)',
    yaxis_title='Side Effect Rate (%)',
    yaxis=dict(range=[0, 100])
)

# Store the figure for display
output_vars['fig1'] = fig1

# Create a grouped bar chart with stratified rates
fig2 = px.bar(
    stratified_rates,
    x='Dosage (mg)',
    y='Side Effect Rate (%)',
    color='Gender',
    barmode='group',
    title='Side Effect Rates Stratified by Gender',
    text_auto='.1f'  # Show percentages on bars
)

# Update layout
fig2.update_layout(
    xaxis_title='Dosage (mg)',
    yaxis_title='Side Effect Rate (%)',
    yaxis=dict(range=[0, 100])
)

# Store the stratified figure
output_vars['fig2'] = fig2

# Create a line plot to better see the trends
fig3 = px.line(
    stratified_rates,
    x='Dosage (mg)',
    y='Side Effect Rate (%)',
    color='Gender',
    markers=True,
    title='Side Effect Rates Trend by Gender'
)

# Add the overall trend
fig3.add_trace(
    go.Scatter(
        x=overall_rates['Dosage (mg)'],
        y=overall_rates['Side Effect Rate (%)'],
        mode='lines+markers',
        name='Overall',
        line=dict(color='black', dash='dash')
    )
)

# Update layout
fig3.update_layout(
    xaxis_title='Dosage (mg)',
    yaxis_title='Side Effect Rate (%)',
    yaxis=dict(range=[0, 100])
)

# Store the trend figure
output_vars['fig3'] = fig3

# Calculate the absolute gender difference at each dose
diff_data = stratified_rates.pivot(index='Dosage (mg)', columns='Gender', values='Side Effect Rate (%)')
diff_data['Absolute Difference'] = diff_data['Female'] - diff_data['Male']
diff_data = diff_data.reset_index()

# Create a plot showing the gender difference
fig4 = px.bar(
    diff_data,
    x='Dosage (mg)',
    y='Absolute Difference',
    title='Absolute Gender Difference in Side Effect Rates',
    text_auto='.1f',
    color='Absolute Difference',
    color_continuous_scale=['blue', 'gray', 'red'],
    color_continuous_midpoint=0
)

# Update layout
fig4.update_layout(
    xaxis_title='Dosage (mg)',
    yaxis_title='Female - Male Difference (%)'
)

# Store the difference figure
output_vars['fig4'] = fig4
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see the results:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Try yourself"):
            st.markdown("""
            **Try modifying:**
            - Adjust the base rates to create different baseline differences
            - Change the dose effects to make the gender difference larger or smaller
            - Try making one gender more sensitive to dose increases
            
            **Challenge yourself:**
            1. What happens if you make the dose effect negative for one gender?
            2. Can you add a third gender category (e.g., 'Other')?
            3. Try adding a statistical test to compare side effect rates between genders
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
            if 'fig1' in output_vars:
                st.plotly_chart(output_vars['fig1'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
                
            if 'fig3' in output_vars:
                st.plotly_chart(output_vars['fig3'], use_container_width=True)
                
            if 'fig4' in output_vars:
                st.plotly_chart(output_vars['fig4'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Insights")
    st.markdown("""
    ### What This Analysis Shows:
    
    1. **Gender Differences in Side Effects**: 
       - The unstratified analysis (first plot) shows the overall side effect rate by dose
       - The stratified analysis (second and third plots) reveals how side effect rates differ between males and females
    
    2. **Why These Differences Matter**:
       - Safety concerns: Higher-risk groups may need more monitoring
       - Dose adjustments: Gender-specific dosing might be appropriate 
       - Patient counseling: Different expectations for side effects
       - Clinical trial design: Ensuring balanced gender representation
    
    3. **Dose-Response Relationships**:
       - The slope of each line indicates how quickly side effects increase with dose
       - Differences in slopes suggest different sensitivities to dose increases
       - The absolute difference plot shows whether the gender gap widens with increasing dose
    
    ### Clinical Applications:
    
    - **Prescribing decisions**: Consider gender as a factor when selecting initial doses
    - **Package inserts**: Include gender-stratified side effect data when relevant
    - **Research prioritization**: Investigate mechanisms behind gender differences
    - **Regulatory considerations**: Gender-specific warnings or contraindications may be needed
    
    This type of stratified analysis is essential for understanding subgroup effects in drug safety!
    """)

def disease_severity_example():
    st.header("Treatment Efficacy by Disease Severity")
    
    st.markdown("""
    ### Analyzing Treatment Efficacy Across Disease Severity Levels
    
    The same treatment might work differently depending on how severe a patient's disease is.
    This example explores how to analyze treatment efficacy stratified by disease severity.
    
    Let's examine this through code:
    """)
    
    # Initial code example
    initial_code = """# Analyzing treatment efficacy stratified by disease severity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# MODIFY THESE PARAMETERS TO SEE DIFFERENT PATTERNS
# Effect size by disease severity (higher = better treatment effect)
mild_effect = 0.8      # Effect in mild disease
moderate_effect = 0.5  # Effect in moderate disease
severe_effect = 0.2    # Effect in severe disease
patients_per_group = 40  # Number of patients per severity group

# Generate synthetic clinical trial data
def generate_trial_data(effect_size, n_patients, severity):
    # Randomly assign patients to treatment (1) or placebo (0)
    treatment = np.random.randint(0, 2, n_patients)
    
    # Generate baseline disease scores (higher = worse disease)
    # Mild: 1-3, Moderate: 4-6, Severe: 7-10
    if severity == 'Mild':
        baseline_score = np.random.uniform(1, 3, n_patients)
    elif severity == 'Moderate':
        baseline_score = np.random.uniform(4, 6, n_patients)
    else:  # Severe
        baseline_score = np.random.uniform(7, 10, n_patients)
    
    # Calculate improvement score based on treatment and effect size
    # Add random noise to simulate individual variation
    improvement = (treatment * effect_size * baseline_score +  
                  np.random.normal(0, 0.5, n_patients))
    
    # Improvement can't be negative (placebo patients might get worse)
    improvement = np.maximum(0, improvement)
    
    # Create dataframe
    return pd.DataFrame({
        'Severity': [severity] * n_patients,
        'Treatment': ['Treatment' if t == 1 else 'Placebo' for t in treatment],
        'Baseline_Score': baseline_score,
        'Improvement': improvement,
        'Percent_Improvement': (improvement / baseline_score) * 100
    })

# Generate data for each severity level
mild_data = generate_trial_data(mild_effect, patients_per_group, 'Mild')
moderate_data = generate_trial_data(moderate_effect, patients_per_group, 'Moderate')
severe_data = generate_trial_data(severe_effect, patients_per_group, 'Severe')

# Combine all data
all_data = pd.concat([mild_data, moderate_data, severe_data], ignore_index=True)

# Print summary
print("Data summary (first 5 rows):")
print(all_data.head())

# Calculate average improvement by treatment group (unstratified)
unstratified_results = all_data.groupby('Treatment')['Percent_Improvement'].agg(['mean', 'sem']).reset_index()
unstratified_results.columns = ['Treatment', 'Mean_Improvement', 'SEM']

print("\\nUnstratified results:")
print(unstratified_results)

# Run t-test for unstratified data
treatment_data = all_data[all_data['Treatment'] == 'Treatment']['Percent_Improvement']
placebo_data = all_data[all_data['Treatment'] == 'Placebo']['Percent_Improvement']
t_stat, p_value = stats.ttest_ind(treatment_data, placebo_data)

print(f"Unstratified t-test: t = {t_stat:.3f}, p = {p_value:.3f}")

# Create unstratified bar chart
fig1 = px.bar(
    unstratified_results,
    x='Treatment',
    y='Mean_Improvement',
    title='Overall Treatment Effect (All Severities Combined)',
    color='Treatment',
    error_y='SEM',
    text_auto='.1f'
)

# Update layout
fig1.update_layout(
    xaxis_title='Treatment Group',
    yaxis_title='Mean Percent Improvement (%)'
)

# Store the figure for display
output_vars['fig1'] = fig1

# Calculate stratified results
stratified_results = all_data.groupby(['Severity', 'Treatment'])['Percent_Improvement'].agg(
    ['mean', 'sem', 'count']
).reset_index()
stratified_results.columns = ['Severity', 'Treatment', 'Mean_Improvement', 'SEM', 'Count']

print("\\nStratified results:")
print(stratified_results)

# Run t-tests for each severity group
p_values = {}
for severity in ['Mild', 'Moderate', 'Severe']:
    severity_data = all_data[all_data['Severity'] == severity]
    treatment_data = severity_data[severity_data['Treatment'] == 'Treatment']['Percent_Improvement']
    placebo_data = severity_data[severity_data['Treatment'] == 'Placebo']['Percent_Improvement']
    
    t_stat, p_val = stats.ttest_ind(treatment_data, placebo_data)
    p_values[severity] = (t_stat, p_val)
    print(f"{severity} severity t-test: t = {t_stat:.3f}, p = {p_val:.3f}")

# Create stratified bar chart
fig2 = px.bar(
    stratified_results,
    x='Severity',
    y='Mean_Improvement',
    color='Treatment',
    barmode='group',
    title='Treatment Effect Stratified by Disease Severity',
    error_y='SEM',
    text_auto='.1f'
)

# Add p-value annotations
for i, severity in enumerate(['Mild', 'Moderate', 'Severe']):
    t_stat, p_val = p_values[severity]
    sig_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    sig_symbol = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    
    fig2.add_annotation(
        x=severity,
        y=stratified_results[(stratified_results['Severity'] == severity) & 
                           (stratified_results['Treatment'] == 'Treatment')]['Mean_Improvement'].values[0] + 5,
        text=sig_symbol,
        showarrow=False,
        font=dict(size=16)
    )

# Update layout
fig2.update_layout(
    xaxis_title='Disease Severity',
    yaxis_title='Mean Percent Improvement (%)'
)

# Store the stratified figure
output_vars['fig2'] = fig2

# Calculate treatment effect size (difference between treatment and placebo)
effect_data = []

for severity in ['Mild', 'Moderate', 'Severe']:
    severity_results = stratified_results[stratified_results['Severity'] == severity]
    treatment_mean = severity_results[severity_results['Treatment'] == 'Treatment']['Mean_Improvement'].values[0]
    placebo_mean = severity_results[severity_results['Treatment'] == 'Placebo']['Mean_Improvement'].values[0]
    
    effect_data.append({
        'Severity': severity,
        'Treatment_Effect': treatment_mean - placebo_mean
    })

effect_df = pd.DataFrame(effect_data)

# Create effect size plot
fig3 = px.bar(
    effect_df,
    x='Severity',
    y='Treatment_Effect',
    title='Treatment Effect Size by Disease Severity',
    color='Treatment_Effect',
    text_auto='.1f',
    color_continuous_scale='RdBu_r'
)

# Update layout
fig3.update_layout(
    xaxis_title='Disease Severity',
    yaxis_title='Treatment Effect (Treatment - Placebo)'
)

# Store the effect size figure
output_vars['fig3'] = fig3

# Create a box plot to show data distribution
fig4 = px.box(
    all_data,
    x='Severity',
    y='Percent_Improvement',
    color='Treatment',
    title='Distribution of Improvement by Severity and Treatment',
    points='all'
)

# Update layout
fig4.update_layout(
    xaxis_title='Disease Severity',
    yaxis_title='Percent Improvement (%)'
)

# Store the box plot
output_vars['fig4'] = fig4
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see the results:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Try yourself"):
            st.markdown("""
            **Try modifying:**
            - Change the effect sizes for different severity levels
            - Try reversing the pattern (better effect in severe disease)
            - Increase the number of patients to see how it affects statistical significance
            
            **Challenge yourself:**
            1. Try adding a fourth severity level (e.g., 'Very Severe')
            2. Modify the code to include a subgroup analysis by age
            3. Calculate and visualize the number needed to treat (NNT) for each severity level
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
            if 'fig1' in output_vars:
                st.plotly_chart(output_vars['fig1'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
                
            if 'fig3' in output_vars:
                st.plotly_chart(output_vars['fig3'], use_container_width=True)
                
            if 'fig4' in output_vars:
                st.plotly_chart(output_vars['fig4'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Insights")
    st.markdown("""
    ### What This Analysis Shows:
    
    1. **Heterogeneity of Treatment Effects**: 
       - The unstratified analysis (first plot) shows the overall treatment effect
       - The stratified analysis (second plot) reveals how treatment efficacy varies by disease severity
    
    2. **Importance for Clinical Decision Making**:
       - Treatment decisions should consider disease severity
       - For some patients, the benefits may not outweigh the risks
       - Resources can be focused on patients most likely to benefit
    
    3. **Statistical Significance vs. Clinical Significance**:
       - A statistically significant result may not be clinically meaningful
       - The effect size plot shows the magnitude of benefit in each group
       - The box plot shows the variability within each group, revealing overlap
    
    ### Clinical Applications:
    
    - **Treatment guidelines**: May recommend different approaches based on severity
    - **Patient counseling**: Setting appropriate expectations for treatment benefits
    - **Resource allocation**: Prioritizing treatment for patients most likely to benefit
    - **Future research**: Focusing on improving outcomes for severity levels with poor response
    
    This type of stratified analysis helps clinicians make more personalized treatment decisions!
    """)

if __name__ == "__main__":
    app()