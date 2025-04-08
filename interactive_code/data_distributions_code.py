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
    st.title("Interactive Coding Laboratory: Data Distributions")
    
    st.markdown("""
    ## Learn by Doing: Apply Health Data Concepts with Code
    
    This interactive section lets you modify code samples to see how different data patterns work. 
    You don't need to be a programmer - just make small changes to the highlighted values and see what happens!
    
    Each example focuses on a common pattern in health data that's relevant to pharmacy practice.
    
    Choose a topic below to get started:
    """)
    
    # Topic selection
    topic = st.selectbox(
        "Select a data pattern to explore:",
        ["Normal Distribution: Patient Measurements", 
         "Yes/No Outcomes: Treatment Response",
         "Rare Events: Adverse Reactions",
         "Time-to-Event: Treatment Response Time",
         "Skewed Data: Lab Values and Concentrations"]
    )
    
    # Display the selected topic content
    if topic == "Normal Distribution: Patient Measurements":
        normal_distribution_lesson()
    elif topic == "Yes/No Outcomes: Treatment Response":
        binomial_distribution_lesson()
    elif topic == "Rare Events: Adverse Reactions":
        poisson_distribution_lesson()
    elif topic == "Time-to-Event: Treatment Response Time":
        exponential_distribution_lesson()
    elif topic == "Skewed Data: Lab Values and Concentrations":
        lognormal_distribution_lesson()

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

def normal_distribution_lesson():
    st.header("Normal Distribution: Patient Measurements")
    
    st.markdown("""
    ###  Normal Distribution (The Bell Curve)
    
    The Normal Distribution (bell curve) is common in health measurements. Patient values tend to cluster around an average, 
    with fewer patients having unusually high or low values.
    
    **Health Examples:**
    - Blood pressure readings
    - Cholesterol levels
    - Patient weights
    - Drug metabolism rates
    - Lab test results
    
    Let's explore blood pressure data to see how it typically follows a (Normal Distribution) bell curve pattern.
    """)
    
    # Initial code example
    initial_code = """# Exploring blood pressure data following a Normal Distribution pattern
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
mean_bp = 120  # average systolic BP in mmHg
std_bp = 15    # spread of BP values in mmHg
sample_size = 500  # number of patients

# Generate simulated blood pressure readings
np.random.seed(42)  # This ensures you get the same results each time
bp_measurements = np.random.normal(mean_bp, std_bp, sample_size)

# Create a DataFrame for the data
data = pd.DataFrame({
    'Systolic BP (mmHg)': bp_measurements
})

# Print summary information
print("Blood Pressure Summary:")
print(f"Average: {np.mean(bp_measurements):.1f} mmHg")
print(f"Middle value (median): {np.median(bp_measurements):.1f} mmHg")
print(f"Spread (standard deviation): {np.std(bp_measurements):.1f} mmHg")
print(f"Lowest value: {np.min(bp_measurements):.1f} mmHg")
print(f"Highest value: {np.max(bp_measurements):.1f} mmHg")

# Create a histogram with normal curve overlay
fig = go.Figure()

# Add histogram of blood pressure values
fig.add_trace(go.Histogram(
    x=bp_measurements,
    histnorm='probability density',
    name='Blood Pressure Data',
    marker_color='lightblue',
    opacity=0.7,
    nbinsx=30
))

# Add the theoretical Normal Distribution
x_values = np.linspace(
    mean_bp - 4*std_bp, 
    mean_bp + 4*std_bp, 
    1000
)
y_values = stats.norm.pdf(x_values, mean_bp, std_bp)

fig.add_trace(go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines',
    name=f'Expected Normal Distribution',
    line=dict(color='red', width=2)
))

# Add reference lines for hypertension categories
bp_categories = [
    {'name': 'Normal', 'threshold': 120, 'color': 'green'},
    {'name': 'Elevated', 'threshold': 130, 'color': 'yellow'},
    {'name': 'Stage 1 Hypertension', 'threshold': 140, 'color': 'orange'},
    {'name': 'Stage 2 Hypertension', 'threshold': 160, 'color': 'red'}
]

for cat in bp_categories:
    # Add vertical line
    fig.add_shape(
        type="line",
        x0=cat['threshold'], y0=0, x1=cat['threshold'], y1=max(y_values) * 0.9,
        line=dict(color=cat['color'], width=2, dash="dash")
    )
    
    # Add label
    fig.add_annotation(
        x=cat['threshold'],
        y=max(y_values) * (0.8 - bp_categories.index(cat) * 0.15),
        text=cat['name'],
        showarrow=False,
        font=dict(color=cat['color'])
    )

# Calculate percentages within standard deviation ranges
within_1sd = np.sum((mean_bp - std_bp <= bp_measurements) & 
                   (bp_measurements <= mean_bp + std_bp)) / sample_size * 100
within_2sd = np.sum((mean_bp - 2*std_bp <= bp_measurements) & 
                   (bp_measurements <= mean_bp + 2*std_bp)) / sample_size * 100
within_3sd = np.sum((mean_bp - 3*std_bp <= bp_measurements) & 
                   (bp_measurements <= mean_bp + 3*std_bp)) / sample_size * 100

print(f"\\nPercentage of patients within:")
print(f"1 standard deviation: {within_1sd:.1f}% (expected: about 68%)")
print(f"2 standard deviations: {within_2sd:.1f}% (expected: about 95%)")
print(f"3 standard deviations: {within_3sd:.1f}% (expected: about 99.7%)")

# Calculate hypertension prevalence based on categories
normal = np.mean(bp_measurements < 120) * 100
elevated = np.mean((bp_measurements >= 120) & (bp_measurements < 130)) * 100
stage1 = np.mean((bp_measurements >= 130) & (bp_measurements < 140)) * 100
stage2 = np.mean(bp_measurements >= 140) * 100

print(f"\\nBlood Pressure Categories:")
print(f"Normal (<120 mmHg): {normal:.1f}% of patients")
print(f"Elevated (120-129 mmHg): {elevated:.1f}% of patients")
print(f"Stage 1 Hypertension (130-139 mmHg): {stage1:.1f}% of patients")
print(f"Stage 2 Hypertension (≥140 mmHg): {stage2:.1f}% of patients")

# Update plot layout
fig.update_layout(
    title='Distribution of Systolic Blood Pressure',
    xaxis_title='Systolic Blood Pressure (mmHg)',
    yaxis_title='Frequency',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

# Store the figure in output_vars for display
output_vars['fig'] = fig
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see what happens:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try changing these values:**
            - Change `mean_bp` to 130 to model a population with higher average BP
            - Change `std_bp` to 10 to see a narrower, more peaked Normal Distribution
            - Increase `sample_size` to 1000 to see how a larger sample affects the curve
            
            **Challenge yourself:**
            1. Change the values to model an elderly population with higher BP
            2. What happens if you set `std_bp` to a very small number like 5?
            3. Can you find values that would indicate 25% of the population has Stage 2 hypertension?
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Results:")
                st.text(output)
            
            # Display any figures generated
            if 'fig' in output_vars:
                st.plotly_chart(output_vars['fig'], use_container_width=True)
        else:
            # Display error message
            st.error("There's an error in the code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Clinical Relevance")
    st.markdown("""
    ### Why Normal Distribution Matters in Pharmacy:
    
    1. **Reference Ranges**: Most laboratory test reference ranges are based on capturing 95% of the healthy population (roughly mean ± 2 standard deviations).
    
    2. **Drug Dosing**: Individual variations in drug metabolism often follow a Normal Distribution. This helps predict what percentage of patients might need dosage adjustments.
    
    3. **Population Health**: Understanding the distribution of health measurements helps target interventions. For example, if you know the distribution of blood pressure in your patient population, you can estimate how many might need medication adjustments.
    
    4. **Clinical Trials**: When designing studies, researchers use the Normal Distribution properties to determine appropriate sample sizes and interpret results.
    
    ### Practical Applications:
    
    - Predicting the percentage of patients who might fall outside normal ranges
    - Understanding what constitutes a "significant" deviation from normal
    - Interpreting laboratory results in the context of population norms
    - Estimating medication needs for a population
    """)

def binomial_distribution_lesson():
    st.header("Yes/No Outcomes: Treatment Response")
    
    st.markdown("""
    ### Yes/No Outcomes (Binomial Distribution)
    
    Many health outcomes are binary - a patient either responds to treatment or doesn't, 
    experiences a side effect or doesn't, etc. When we count these yes/no outcomes across 
    multiple patients, the pattern follows what's called a binomial distribution.
    
    **Health Examples:**
    - Number of patients responding to a medication
    - Patients experiencing a specific side effect
    - Vaccine success or failure
    - Medication adherence (took medication vs. didn't take it)
    
    Let's explore treatment response data to see this pattern.
    """)
    
    # Initial code example
    initial_code = """# Exploring treatment response data (yes/no outcomes)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_patients = 50  # Number of patients in each treatment group
p_control = 0.30   # Probability of response in control group (30%)
p_treatment = 0.60   # Probability of response in treatment group (60%)
n_trials = 1000  # Number of simulated clinical trials

# Simulate the control group
np.random.seed(42)  # For consistent results
control_responses = np.random.binomial(n_patients, p_control, n_trials)

# Simulate the treatment group
treatment_responses = np.random.binomial(n_patients, p_treatment, n_trials)

# Calculate response rates for each group
control_rates = control_responses / n_patients * 100
treatment_rates = treatment_responses / n_patients * 100

# Calculate absolute risk reduction and number needed to treat
absolute_risk_reduction = treatment_rates - control_rates
number_needed_to_treat = 100 / absolute_risk_reduction

# Create a DataFrame with the results
data = pd.DataFrame({
    'Control Responses': control_responses,
    'Treatment Responses': treatment_responses,
    'Control Response Rate (%)': control_rates,
    'Treatment Response Rate (%)': treatment_rates,
    'Absolute Risk Reduction (%)': absolute_risk_reduction,
    'Number Needed to Treat': number_needed_to_treat
})

# Print summary results
print("Treatment Response Summary:")
print(f"Average Control Responses: {np.mean(control_responses):.1f} out of {n_patients} patients")
print(f"Average Treatment Responses: {np.mean(treatment_responses):.1f} out of {n_patients} patients")
print(f"Average Control Response Rate: {np.mean(control_rates):.1f}%")
print(f"Average Treatment Response Rate: {np.mean(treatment_rates):.1f}%")
print(f"Average Absolute Risk Reduction: {np.mean(absolute_risk_reduction):.1f}%")
print(f"Average Number Needed to Treat: {np.median(number_needed_to_treat):.1f} patients")

# Create plot comparing response distributions
fig = go.Figure()

# Control group histogram
fig.add_trace(go.Histogram(
    x=control_responses,
    name='Control Group',
    opacity=0.7,
    marker_color='lightblue',
    nbinsx=n_patients//2
))

# Treatment group histogram
fig.add_trace(go.Histogram(
    x=treatment_responses,
    name='Treatment Group',
    opacity=0.7,
    marker_color='lightgreen',
    nbinsx=n_patients//2
))

# Add vertical lines for expected values
fig.add_shape(
    type="line",
    x0=n_patients * p_control, y0=0, x1=n_patients * p_control, y1=120,
    line=dict(color="blue", width=2, dash="dash")
)

fig.add_shape(
    type="line",
    x0=n_patients * p_treatment, y0=0, x1=n_patients * p_treatment, y1=120,
    line=dict(color="green", width=2, dash="dash")
)

# Add annotations
fig.add_annotation(
    x=n_patients * p_control,
    y=100,
    text=f"Expected: {n_patients * p_control:.1f}",
    showarrow=True,
    arrowhead=1,
    ax=-40,
    ay=0,
    font=dict(color="blue")
)

fig.add_annotation(
    x=n_patients * p_treatment,
    y=100,
    text=f"Expected: {n_patients * p_treatment:.1f}",
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=0,
    font=dict(color="green")
)

# Update layout
fig.update_layout(
    title='Distribution of Patient Responses Across Clinical Trials',
    xaxis_title='Number of Patients Responding',
    yaxis_title='Number of Trials',
    barmode='overlay',
    template="plotly_white"
)

output_vars['fig'] = fig

# Create a second plot for NNT distribution
fig2 = go.Figure()

# Filter out extreme NNT values for better visualization
filtered_nnt = number_needed_to_treat[number_needed_to_treat < 100]

# Add NNT histogram
fig2.add_trace(go.Histogram(
    x=filtered_nnt,
    marker_color='purple',
    opacity=0.7,
    nbinsx=20
))

# Add vertical line for median NNT
median_nnt = np.median(filtered_nnt)
fig2.add_shape(
    type="line",
    x0=median_nnt, y0=0, x1=median_nnt, y1=100,
    line=dict(color="red", width=2, dash="dash")
)

fig2.add_annotation(
    x=median_nnt,
    y=80,
    text=f"Median NNT: {median_nnt:.1f}",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-40,
    font=dict(color="red")
)

# Update layout
fig2.update_layout(
    title='Distribution of Number Needed to Treat (NNT)',
    xaxis_title='Number Needed to Treat',
    yaxis_title='Frequency',
    template="plotly_white"
)

output_vars['fig2'] = fig2

# Calculate probability of observing clinically meaningful difference
meaningful_difference = 5  # At least 5 more responses in treatment vs control
probability = np.mean((treatment_responses - control_responses) >= meaningful_difference) * 100

print(f"\\nProbability of observing at least {meaningful_difference} more responses")
print(f"in the treatment group compared to control: {probability:.1f}%")

# Calculate confidence interval for response rate difference
ci_lower = np.percentile(treatment_rates - control_rates, 2.5)
ci_upper = np.percentile(treatment_rates - control_rates, 97.5)
print(f"\\n95% Confidence interval for response rate difference: {ci_lower:.1f}% to {ci_upper:.1f}%")
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see what happens:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try changing these values:**
            - Change `p_treatment` to 0.40 to see what happens with a smaller effect
            - Increase `n_patients` to 100 to see how larger sample sizes affect results
            - Change `meaningful_difference` to see how it affects the probability
            
            **Challenge yourself:**
            1. What happens if you make both response rates the same (e.g., both 0.50)?
            2. How large does your sample size need to be to reliably detect a 10% difference?
            3. What happens to the Number Needed to Treat when the treatment effect is small?
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Results:")
                st.text(output)
            
            # Display any figures generated
            if 'fig' in output_vars:
                st.plotly_chart(output_vars['fig'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
        else:
            # Display error message
            st.error("There's an error in the code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Clinical Relevance")
    st.markdown("""
    ### Why Yes/No Outcomes Matter in Pharmacy:
    
    1. **Interpreting Clinical Trials**: Most clinical trials report binary outcomes like "responded to treatment" or "experienced side effect." Understanding how these outcomes vary across patients helps interpret study results.
    
    2. **Number Needed to Treat (NNT)**: This practical measure tells you how many patients you need to treat to prevent one bad outcome or achieve one good outcome. Lower NNT values indicate more effective treatments.
    
    3. **Patient Counseling**: Understanding the probability of benefit or harm helps provide patients with realistic expectations about their treatments.
    
    4. **Sample Size Considerations**: When designing studies, researchers need to account for the expected response rates to ensure the study has enough participants.
    
    ### Practical Applications:
    
    - Evaluating new medications by comparing response rates
    - Estimating the likelihood of side effects
    - Deciding between treatment options based on NNT
    - Predicting outcomes for a group of patients with similar characteristics
    """)

def poisson_distribution_lesson():
    st.header("Rare Events: Adverse Reactions")
    
    st.markdown("""
    ### Rare Event Pattern (Poisson Distribution)
    
    Some health events occur rarely but relatively consistently over time, like medication errors
    or adverse drug reactions. When counting these types of rare events, we often see a pattern
    called the Poisson distribution.
    
    **Health Examples:**
    - Adverse drug reactions per month
    - Medication errors in a pharmacy
    - Hospital admissions for drug toxicity
    - Rare disease cases in a region
    - Serious side effects during a clinical trial
    
    Let's explore adverse reaction data to see this pattern.
    """)
    
    # Initial code example
    initial_code = """# Exploring adverse drug reaction data
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
avg_reactions = 2.5  # Average number of reported adverse reactions per week
weeks = 52         # Number of weeks to monitor (1 year)
alert_threshold = 6  # Number of reactions that would trigger an investigation

# Simulate weekly adverse reaction counts for a year
np.random.seed(42)  # For consistent results
weekly_reactions = np.random.poisson(avg_reactions, weeks)

# Create a time series dataframe
dates = pd.date_range(start='2023-01-01', periods=weeks, freq='W')
time_series = pd.DataFrame({
    'Date': dates,
    'Adverse Reactions': weekly_reactions,
    'Month': [d.month_name() for d in dates],
    'Alert Triggered': weekly_reactions >= alert_threshold
})

# Print summary statistics
print("Adverse Reaction Summary:")
print(f"Average reactions per week: {np.mean(weekly_reactions):.1f} (expected: {avg_reactions:.1f})")
print(f"Most reactions in a single week: {np.max(weekly_reactions)}")
print(f"Weeks with zero reactions: {np.sum(weekly_reactions == 0)}")
print(f"Total annual reactions: {np.sum(weekly_reactions)}")
print(f"Number of weeks exceeding alert threshold: {np.sum(weekly_reactions >= alert_threshold)}")

# Calculate theoretical probabilities
k_values = np.arange(0, 15)  # 0 to 14 reactions
poisson_pmf = stats.poisson.pmf(k_values, avg_reactions)

# Create bar chart of observed vs. expected frequencies
observed_counts = np.bincount(weekly_reactions, minlength=15)
observed_freq = observed_counts / weeks

# Create comparison plot
fig = go.Figure()

# Add bars for observed frequencies
fig.add_trace(go.Bar(
    x=k_values,
    y=observed_freq[:15],
    name='Observed Frequency',
    marker_color='skyblue',
    opacity=0.7
))

# Add bars for theoretical probabilities
fig.add_trace(go.Bar(
    x=k_values,
    y=poisson_pmf,
    name='Expected Probability',
    marker_color='red',
    opacity=0.5
))

# Add vertical line for alert threshold
fig.add_shape(
    type="line",
    x0=alert_threshold, y0=0, x1=alert_threshold, y1=0.3,
    line=dict(color="red", width=2, dash="dash")
)

fig.add_annotation(
    x=alert_threshold,
    y=0.25,
    text=f"Alert Threshold: {alert_threshold}",
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=0,
    font=dict(color="red")
)

# Update layout
fig.update_layout(
    title='Adverse Reaction Counts: Observed vs. Expected',
    xaxis_title='Number of Adverse Reactions per Week',
    yaxis_title='Probability/Frequency',
    barmode='group',
    template="plotly_white"
)

output_vars['fig'] = fig

# Create a time series plot
fig2 = px.line(
    time_series,
    x='Date',
    y='Adverse Reactions',
    title='Weekly Adverse Reaction Counts',
    template="plotly_white"
)

# Add horizontal line for the average rate
fig2.add_shape(
    type="line",
    x0=min(dates), y0=avg_reactions, x1=max(dates), y1=avg_reactions,
    line=dict(color="blue", width=2, dash="dash")
)

# Add horizontal line for the alert threshold
fig2.add_shape(
    type="line",
    x0=min(dates), y0=alert_threshold, x1=max(dates), y1=alert_threshold,
    line=dict(color="red", width=2, dash="dash")
)

fig2.add_annotation(
    x=dates[weeks//4],
    y=avg_reactions * 1.2,
    text=f"Average Rate: {avg_reactions}",
    showarrow=False,
    font=dict(color="blue")
)

fig2.add_annotation(
    x=dates[weeks//4],
    y=alert_threshold * 1.1,
    text=f"Alert Threshold: {alert_threshold}",
    showarrow=False,
    font=dict(color="red")
)

# Mark alert weeks
alert_weeks = time_series[time_series['Alert Triggered']]
if len(alert_weeks) > 0:
    fig2.add_trace(go.Scatter(
        x=alert_weeks['Date'],
        y=alert_weeks['Adverse Reactions'],
        mode='markers',
        name='Alert Triggered',
        marker=dict(color='red', size=10, symbol='x')
    ))

output_vars['fig2'] = fig2

# Calculate probability of exceeding alert threshold
prob_exceeding = 1 - stats.poisson.cdf(alert_threshold - 1, avg_reactions)
expected_alerts = prob_exceeding * weeks
print(f"\\nProbability of exceeding alert threshold in any week: {prob_exceeding:.1%}")
print(f"Expected number of alerts per year: {expected_alerts:.1f}")

# Calculate probability of having a week with zero reactions
prob_zero = stats.poisson.pmf(0, avg_reactions)
print(f"Probability of having zero reactions in a week: {prob_zero:.1%}")
print(f"Expected number of zero-reaction weeks per year: {prob_zero * weeks:.1f}")

# Calculate monthly totals
monthly_data = time_series.groupby('Month')['Adverse Reactions'].sum().reset_index()

# Create monthly totals bar chart
fig3 = px.bar(
    monthly_data,
    x='Month',
    y='Adverse Reactions',
    title='Monthly Adverse Reaction Totals',
    template="plotly_white",
    color='Adverse Reactions',
    color_continuous_scale='blues'
)

# Calculate expected monthly total
expected_monthly = avg_reactions * 4.33  # average number of weeks in a month

# Add horizontal line for expected monthly total
fig3.add_shape(
    type="line",
    x0=-0.5, y0=expected_monthly, x1=11.5, y1=expected_monthly,
    line=dict(color="red", width=2, dash="dash")
)

fig3.add_annotation(
    x=6,
    y=expected_monthly * 1.1,
    text=f"Expected Monthly Total: {expected_monthly:.1f}",
    showarrow=False,
    font=dict(color="red")
)

output_vars['fig3'] = fig3
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see what happens:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try changing these values:**
            - Change `avg_reactions` to 1.0 to model a rarer adverse reaction
            - Change `alert_threshold` to 5 or 7 to see how it affects alert frequency
            - Increase `weeks` to 104 (2 years) to see patterns over a longer period
            
            **Challenge yourself:**
            1. What happens when avg_reactions gets very small (like 0.5)? 
            2. How would you set the alert threshold to trigger an investigation about once per quarter?
            3. What happens if you make avg_reactions and alert_threshold equal?
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Results:")
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
            st.error("There's an error in the code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Clinical Relevance")
    st.markdown("""
    ### Why Rare Event Patterns Matter in Pharmacy:
    
    1. **Pharmacovigilance**: Monitoring and detecting unusual patterns in adverse drug reactions is crucial for patient safety. Understanding the expected random variation helps distinguish between normal fluctuations and genuine safety signals.
    
    2. **Setting Alert Thresholds**: Facilities need to set appropriate thresholds for when to investigate clusters of adverse events. Too low, and you waste resources on false alarms; too high, and you might miss important signals.
    
    3. **Resource Planning**: Knowing the expected pattern of rare events helps pharmacies and hospitals allocate appropriate resources for handling them.
    
    4. **Regulatory Reporting**: Understanding when an observed rate of adverse events exceeds the expected background rate is essential for regulatory reporting.
    
    ### Practical Applications:
    
    - Monitoring adverse drug reaction reports
    - Setting up surveillance systems for medication errors
    - Planning staffing based on expected admission rates
    - Analyzing rare but serious side effects in post-marketing surveillance
    - Evaluating quality improvement initiatives by tracking error rates
    """)

def exponential_distribution_lesson():
    st.header("Time-to-Event: Treatment Response Time")
    
    st.markdown("""
    ### Time-to-Event Pattern (Exponential Distribution)
    
    Many healthcare events involve waiting for something to happen - a patient responding to treatment,
    a side effect occurring, or recovery from an illness. When the risk of the event occurring
    remains constant over time, the waiting times often follow an exponential distribution.
    
    **Health Examples:**
    - Time until pain relief after medication
    - Duration of hospital stays
    - Time until disease recurrence
    - Shelf life of medications
    - Time between adverse events
    
    Let's explore response time data to see this pattern.
    """)
    
    # Initial code example
    initial_code = """# Exploring time-to-response data for pain medication
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
avg_response_time = 20  # Average minutes until pain relief occurs
n_patients = 200     # Number of patients to simulate
max_observation = 60  # Maximum observation time (minutes)

# Simulate response times (minutes until pain relief)
np.random.seed(42)  # For consistent results
true_response_times = np.random.exponential(avg_response_time, n_patients)

# Apply censoring (some patients don't experience relief during observation)
observed_times = np.minimum(true_response_times, max_observation)
censored = observed_times >= max_observation
n_censored = np.sum(censored)

print("Pain Relief Response Time Summary:")
print(f"Number of patients: {n_patients}")
print(f"Average response time: {avg_response_time} minutes")
print(f"Patients experiencing relief during observation: {n_patients - n_censored}")
print(f"Patients not experiencing relief (censored): {n_censored} ({n_censored/n_patients:.1%})")

# Create a DataFrame for the data
data = pd.DataFrame({
    'Patient': np.arange(1, n_patients + 1),
    'Response Time (min)': observed_times,
    'Experienced Relief': ~censored
})

# Calculate descriptive statistics (only for patients who experienced relief)
relief_times = observed_times[~censored]
print(f"\\nFor patients experiencing relief:")
print(f"Minimum time: {np.min(relief_times):.1f} minutes")
print(f"Median time: {np.median(relief_times):.1f} minutes")
print(f"Average time: {np.mean(relief_times):.1f} minutes")
print(f"Maximum time: {np.max(relief_times):.1f} minutes")

# Calculate Kaplan-Meier survival curve
# Sort data by time
sorted_indices = np.argsort(observed_times)
sorted_times = observed_times[sorted_indices]
sorted_events = ~censored[sorted_indices]

# Calculate survival probability (probability of NOT experiencing relief yet)
unique_times = np.unique(sorted_times[sorted_events])
survival_prob = np.ones(len(unique_times) + 1)
times_for_plot = np.concatenate(([0], unique_times))

for i, t in enumerate(unique_times):
    # Number at risk at time t
    n_risk = np.sum(sorted_times >= t)
    # Number of events at time t
    n_events = np.sum((sorted_times == t) & sorted_events)
    # Calculate survival probability
    survival_prob[i+1] = survival_prob[i] * (1 - n_events / n_risk)

# Create plot for Kaplan-Meier curve
fig = go.Figure()

# Add Kaplan-Meier curve
fig.add_trace(go.Scatter(
    x=times_for_plot,
    y=survival_prob,
    mode='lines',
    name='Observed Data',
    line=dict(color='blue', width=2)
))

# Add theoretical survival curve
t_range = np.linspace(0, max_observation, 100)
theoretical_survival = np.exp(-t_range / avg_response_time)

fig.add_trace(go.Scatter(
    x=t_range,
    y=theoretical_survival,
    mode='lines',
    name='Expected Pattern',
    line=dict(color='red', width=2, dash='dash')
))

# Add markers for censored observations
if n_censored > 0:
    fig.add_trace(go.Scatter(
        x=[max_observation] * n_censored,
        y=[survival_prob[-1]] * n_censored,
        mode='markers',
        name='Patients Without Relief',
        marker=dict(color='green', symbol='x', size=8)
    ))

# Add annotations for key time points
# Median response time
median_time = avg_response_time * np.log(2)  # Theoretical median
fig.add_shape(
    type="line",
    x0=median_time, y0=0, x1=median_time, y1=0.5,
    line=dict(color="gray", width=1, dash="dash")
)

fig.add_shape(
    type="line",
    x0=0, y0=0.5, x1=median_time, y1=0.5,
    line=dict(color="gray", width=1, dash="dash")
)

fig.add_annotation(
    x=median_time,
    y=0.52,
    text=f"Median: {median_time:.1f} min",
    showarrow=False,
    font=dict(color="black")
)

# Update layout
fig.update_layout(
    title='Time to Pain Relief',
    xaxis_title='Time (minutes)',
    yaxis_title='Proportion WITHOUT Relief',
    yaxis=dict(range=[0, 1.05]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

output_vars['fig'] = fig

# Create histogram of response times
fig2 = go.Figure()

# Add histogram of observed times (only for those who experienced relief)
fig2.add_trace(go.Histogram(
    x=relief_times,
    histnorm='probability density',
    name='Observed Response Times',
    marker_color='lightblue',
    opacity=0.7,
    nbinsx=20
))

# Add theoretical PDF
fig2.add_trace(go.Scatter(
    x=t_range,
    y=(1/avg_response_time) * np.exp(-t_range/avg_response_time),
    mode='lines',
    name='Expected Pattern',
    line=dict(color='red', width=2)
))

# Update layout
fig2.update_layout(
    title='Distribution of Pain Relief Times',
    xaxis_title='Time (minutes)',
    yaxis_title='Frequency',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

output_vars['fig2'] = fig2

# Calculate clinical response rates at different time points
time_points = [10, 20, 30, 45, 60]
response_rates = []

for t in time_points:
    # Theoretical response rate: 1 - e^(-t/avg)
    theoretical_rate = (1 - np.exp(-t/avg_response_time)) * 100
    # Observed response rate
    observed_rate = np.mean(observed_times <= t) * 100
    response_rates.append({
        'Time (min)': t,
        'Theoretical Response Rate (%)': theoretical_rate,
        'Observed Response Rate (%)': observed_rate
    })

# Create a DataFrame with the response rates
response_df = pd.DataFrame(response_rates)

# Print the response rates
print("\\nResponse Rates at Different Time Points:")
for _, row in response_df.iterrows():
    print(f"At {row['Time (min)']} minutes: {row['Observed Response Rate (%)']:.1f}% (expected: {row['Theoretical Response Rate (%)']:.1f}%)")

# Create a bar chart for response rates
fig3 = go.Figure()

fig3.add_trace(go.Bar(
    x=response_df['Time (min)'],
    y=response_df['Theoretical Response Rate (%)'],
    name='Expected',
    marker_color='red',
    opacity=0.7
))

fig3.add_trace(go.Bar(
    x=response_df['Time (min)'],
    y=response_df['Observed Response Rate (%)'],
    name='Observed',
    marker_color='blue',
    opacity=0.7
))

# Update layout
fig3.update_layout(
    title='Pain Relief Response Rates Over Time',
    xaxis_title='Time (minutes)',
    yaxis_title='Percentage of Patients with Relief',
    yaxis=dict(range=[0, 100]),
    barmode='group',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

output_vars['fig3'] = fig3
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see what happens:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try changing these values:**
            - Change `avg_response_time` to 10 to model a faster-acting medication
            - Change `max_observation` to 30 to see how shorter observation affects results
            - Increase `n_patients` to 500 to see how sample size affects the curves
            
            **Challenge yourself:**
            1. How would the results change if the average response time was 40 minutes?
            2. What happens if you make the observation time shorter than the average response time?
            3. Could you adjust the time points in the response rate chart to better inform patients?
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Results:")
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
            st.error("There's an error in the code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Clinical Relevance")
    st.markdown("""
    ### Why Time-to-Event Patterns Matter in Pharmacy:
    
    1. **Patient Counseling**: Understanding response time distributions helps pharmacists provide realistic expectations to patients about when medications will start working.
    
    2. **Medication Comparison**: Comparing the response time distributions of different medications helps healthcare providers select the most appropriate option for a patient's needs.
    
    3. **Clinical Trial Design**: When designing studies, researchers need to determine appropriate observation periods based on expected response times.
    
    4. **Medication Persistence**: Understanding how long patients typically continue therapy before discontinuing helps identify intervention points.
    
    ### Practical Applications:
    
    - Providing patients with time-to-relief expectations for pain medications
    - Predicting hospital discharge patterns for capacity planning
    - Estimating medication shelf-life
    - Planning appropriate follow-up intervals after treatment initiation
    - Modeling drug refill patterns
    """)

def lognormal_distribution_lesson():
    st.header("Skewed Data: Lab Values and Concentrations")
    
    st.markdown("""
    ### Skewed Data Pattern (Log-Normal Distribution)
    
    Many biological measurements and concentrations show a pattern where most values are on the lower end,
    but a few are much higher (right-skewed). Often, taking the logarithm of these values produces a Normal Distribution,
    which is why we call this pattern "log-normal."
    
    **Health Examples:**
    - Drug concentrations in blood
    - Cholesterol levels
    - Liver enzyme measurements
    - Bacterial colony counts
    - Healthcare costs
    
    Let's explore lab data to see this pattern.
    """)
    
    # Initial code example
    initial_code = """# Exploring skewed lab data (ALT liver enzyme values)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
median_value = 20    # Median ALT value (IU/L)
skewness = 0.6       # Higher values make the distribution more skewed
n_patients = 1000    # Number of patients to simulate
ref_upper_limit = 40 # Upper limit of normal range (IU/L)

# Convert to log-normal parameters
mu = np.log(median_value)  # Log-space mean
sigma = skewness           # Log-space standard deviation

# Calculate expected statistics in original scale
expected_mean = np.exp(mu + sigma**2/2)
expected_median = np.exp(mu)
expected_mode = np.exp(mu - sigma**2)
expected_sd = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))

# Generate log-normally distributed ALT values
np.random.seed(42)  # For consistent results
alt_values = np.random.lognormal(mu, sigma, n_patients)

# Print summary statistics
print("ALT Liver Enzyme Summary:")
print(f"Parameters: median = {median_value}, skewness = {skewness}")
print("\\nExpected Statistics:")
print(f"Mean: {expected_mean:.1f} IU/L")
print(f"Median: {expected_median:.1f} IU/L")
print(f"Mode (most common value): {expected_mode:.1f} IU/L")
print(f"Standard Deviation: {expected_sd:.1f} IU/L")

print("\\nObserved Sample Statistics:")
print(f"Mean: {np.mean(alt_values):.1f} IU/L")
print(f"Median: {np.median(alt_values):.1f} IU/L")
print(f"Standard Deviation: {np.std(alt_values):.1f} IU/L")
print(f"Minimum: {np.min(alt_values):.1f} IU/L")
print(f"Maximum: {np.max(alt_values):.1f} IU/L")

# Calculate percentage above reference range
pct_above_ref = np.mean(alt_values > ref_upper_limit) * 100
print(f"\\nPercentage above reference limit ({ref_upper_limit} IU/L): {pct_above_ref:.1f}%")

# Calculate percentiles
percentiles = [5, 25, 50, 75, 90, 95, 97.5, 99]
percentile_values = np.percentile(alt_values, percentiles)

print("\\nPercentile Values:")
for p, val in zip(percentiles, percentile_values):
    print(f"{p}th percentile: {val:.1f} IU/L")

# Create a DataFrame for the data
data = pd.DataFrame({
    'ALT (IU/L)': alt_values,
    'Log(ALT)': np.log(alt_values),
    'Above Reference': alt_values > ref_upper_limit
})

# Create histogram with skewed distribution
fig = go.Figure()

# Add histogram
fig.add_trace(go.Histogram(
    x=alt_values,
    histnorm='probability density',
    name='Observed ALT Values',
    marker_color='lightblue',
    opacity=0.7,
    nbinsx=30
))

# Generate theoretical PDF for log-normal distribution
x_range = np.linspace(0, np.percentile(alt_values, 99), 1000)  # Limit x-range to 99th percentile for better visualization
lognorm_pdf = stats.lognorm.pdf(x_range, s=sigma, scale=np.exp(mu))

# Add theoretical PDF curve
fig.add_trace(go.Scatter(
    x=x_range,
    y=lognorm_pdf,
    mode='lines',
    name='Expected Pattern',
    line=dict(color='red', width=2)
))

# Add vertical lines for mean, median, and reference limit
fig.add_shape(
    type="line",
    x0=expected_mean, y0=0, x1=expected_mean, y1=max(lognorm_pdf) * 0.9,
    line=dict(color="green", width=2, dash="dash")
)

fig.add_shape(
    type="line",
    x0=expected_median, y0=0, x1=expected_median, y1=max(lognorm_pdf) * 0.8,
    line=dict(color="blue", width=2, dash="dash")
)

fig.add_shape(
    type="line",
    x0=ref_upper_limit, y0=0, x1=ref_upper_limit, y1=max(lognorm_pdf) * 0.7,
    line=dict(color="red", width=2, dash="dash")
)

# Add annotations for key statistics
fig.add_annotation(
    x=expected_mean,
    y=max(lognorm_pdf) * 0.95,
    text=f"Mean: {expected_mean:.1f}",
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=0,
    font=dict(color="green")
)

fig.add_annotation(
    x=expected_median,
    y=max(lognorm_pdf) * 0.85,
    text=f"Median: {expected_median:.1f}",
    showarrow=True,
    arrowhead=1,
    ax=-40,
    ay=0,
    font=dict(color="blue")
)

fig.add_annotation(
    x=ref_upper_limit,
    y=max(lognorm_pdf) * 0.75,
    text=f"Reference Limit: {ref_upper_limit}",
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=0,
    font=dict(color="red")
)

# Update layout
fig.update_layout(
    title='Distribution of ALT Liver Enzyme Values',
    xaxis_title='ALT (IU/L)',
    yaxis_title='Frequency',
    xaxis=dict(range=[0, min(300, np.percentile(alt_values, 99.5))]),  # Limit x-axis for better visualization
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

output_vars['fig'] = fig

# Create histogram of log-transformed values
fig2 = go.Figure()

# Add histogram of log-transformed values
fig2.add_trace(go.Histogram(
    x=np.log(alt_values),
    histnorm='probability density',
    name='Log-Transformed ALT',
    marker_color='lightgreen',
    opacity=0.7,
    nbinsx=30
))

# Generate theoretical normal PDF for log values
log_x_range = np.linspace(np.min(np.log(alt_values)), np.max(np.log(alt_values)), 1000)
normal_pdf = stats.norm.pdf(log_x_range, mu, sigma)

# Add theoretical PDF curve
fig2.add_trace(go.Scatter(
    x=log_x_range,
    y=normal_pdf,
    mode='lines',
    name='Normal Distribution',
    line=dict(color='red', width=2)
))

# Update layout
fig2.update_layout(
    title='Log-Transformed ALT Values (Should be Normally Distributed)',
    xaxis_title='Log(ALT)',
    yaxis_title='Frequency',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

output_vars['fig2'] = fig2

# Divide patients into age groups for comparison
np.random.seed(43)  # Different seed for independent grouping
age_groups = np.random.choice(['18-30', '31-50', '51-70', '70+'], size=n_patients, p=[0.2, 0.3, 0.3, 0.2])

# Create more pronounced differences between age groups
age_factors = {
    '18-30': 0.8,   # Younger patients have lower values
    '31-50': 1.0,   # Reference group
    '51-70': 1.3,   # Older patients have higher values
    '70+': 1.5      # Elderly have highest values
}

# Apply age factors to ALT values
scaled_alt = np.array([alt * age_factors[age] for alt, age in zip(alt_values, age_groups)])

# Create DataFrame with age groups
age_data = pd.DataFrame({
    'Age Group': age_groups,
    'ALT (IU/L)': scaled_alt,
    'Above Reference': scaled_alt > ref_upper_limit
})

# Calculate summary statistics by age group
age_summary = age_data.groupby('Age Group').agg(
    Mean_ALT=('ALT (IU/L)', 'mean'),
    Median_ALT=('ALT (IU/L)', 'median'),
    Pct_Above_Ref=('Above Reference', lambda x: x.mean() * 100)
).reset_index()

print("\\nALT Values by Age Group:")
for _, row in age_summary.iterrows():
    print(f"{row['Age Group']}: Mean = {row['Mean_ALT']:.1f}, " +
          f"Median = {row['Median_ALT']:.1f}, " +
          f"Above Ref: {row['Pct_Above_Ref']:.1f}%")

# Create box plot by age group
fig3 = px.box(
    age_data,
    x='Age Group',
    y='ALT (IU/L)',
    color='Age Group',
    title='ALT Values by Age Group',
    template="plotly_white"
)

# Add reference line
fig3.add_shape(
    type="line",
    x0=-0.5, y0=ref_upper_limit, x1=3.5, y1=ref_upper_limit,
    line=dict(color="red", width=2, dash="dash")
)

fig3.add_annotation(
    x=1.5,
    y=ref_upper_limit * 1.1,
    text=f"Reference Limit: {ref_upper_limit}",
    showarrow=False,
    font=dict(color="red")
)

# Limit y-axis for better visualization
fig3.update_yaxes(range=[0, min(150, np.percentile(scaled_alt, 99))])

output_vars['fig3'] = fig3
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and hit 'Run Code' to see what happens:", 
                            value=initial_code, height=400)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Execute button
        execute_button = st.button("Run Code")
    
    with col2:
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try changing these values:**
            - Increase `skewness` to 1.0 to see a more skewed distribution
            - Change `median_value` to 30 to model a population with higher ALT levels
            - Change `ref_upper_limit` to see how it affects the percentage of abnormal results
            
            **Challenge yourself:**
            1. Try adjusting the parameters to model a population where about 20% have abnormal results
            2. How does changing the skewness affect the difference between mean and median?
            3. What happens to the age group differences if you change the age_factors?
            """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Results:")
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
            st.error("There's an error in the code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Clinical Relevance")
    st.markdown("""
    ### Why Skewed Data Patterns Matter in Pharmacy:
    
    1. **Reference Ranges**: Many laboratory parameters follow log-normal distributions. Understanding this helps interpret reference ranges and patient results properly.
    
    2. **Pharmacokinetics**: Drug concentrations in blood often follow log-normal distributions due to biological processes. This affects dosing strategies and therapeutic drug monitoring.
    
    3. **Data Analysis**: When analyzing skewed laboratory data, transformations (like taking the logarithm) may be necessary before applying certain statistical tests.
    
    4. **Population Comparisons**: When comparing lab values between different populations or age groups, understanding the underlying distribution helps interpret differences correctly.
    
    ### Practical Applications:
    
    - Interpreting liver function tests and other laboratory parameters
    - Setting appropriate reference ranges for different patient populations
    - Analyzing drug concentration data in therapeutic drug monitoring
    - Understanding why mean and median values can differ substantially
    - Identifying truly abnormal values versus expected variation in the upper tail
    """)

if __name__ == "__main__":
    app()