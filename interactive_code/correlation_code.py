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
    st.title("Interactive Coding Laboratory: Correlation Analysis")
    
    st.markdown("""
    ## Learn by Coding: Correlation Concepts
    
    This interactive coding laboratory allows you to modify and execute Python code directly in your browser.
    Experiment with different aspects of correlation by modifying the example code and seeing the results.
    
    Choose a topic to explore:
    """)
    
    # Topic selection
    topic = st.selectbox(
        "Select a correlation topic:",
        ["Basic Correlation Analysis", 
         "Correlation vs. Causation",
         "Types of Correlation",
         "Public Health Correlation Examples"]
    )
    
    # Display the selected topic
    if topic == "Basic Correlation Analysis":
        basic_correlation_lesson()
    elif topic == "Correlation vs. Causation":
        correlation_causation_lesson()
    elif topic == "Types of Correlation":
        correlation_types_lesson()
    elif topic == "Public Health Correlation Examples":
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

def basic_correlation_lesson():
    st.header("Basic Correlation Analysis")
    
    st.markdown("""
    ### Understanding Correlation Coefficients
    
    Correlation measures the strength and direction of the relationship between two variables:
    
    - **+1**: Perfect positive correlation (as one increases, the other increases)
    - **0**: No linear correlation
    - **-1**: Perfect negative correlation (as one increases, the other decreases)
    
    Let's explore how to calculate and visualize correlation:
    """)
    
    # Initial code example
    initial_code = """# Basic correlation analysis
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
correlation = 0.7  # Try values between -1 and 1
n_samples = 100    # Number of data points to generate
noise_level = 0.2  # How much random noise to add

# Generate two correlated variables
np.random.seed(42)  # For reproducible results
x = np.random.normal(0, 1, n_samples)
y = correlation * x + np.random.normal(0, noise_level, n_samples)

# Create a DataFrame with the data
data = pd.DataFrame({
    'Variable 1': x,
    'Variable 2': y
})

# Calculate the Pearson correlation coefficient
pearson_r, p_value = stats.pearsonr(x, y)

# Print results
print(f"Generated data with {n_samples} samples")
print(f"Target correlation: {correlation:.2f}")
print(f"Actual correlation: {pearson_r:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("The correlation is statistically significant (p < 0.05)")
else:
    print("The correlation is not statistically significant (p >= 0.05)")

# Create a scatter plot
fig = px.scatter(
    data, 
    x='Variable 1', 
    y='Variable 2',
    title=f"Scatter Plot (r = {pearson_r:.2f})",
    trendline='ols'  # Add ordinary least squares regression line
)

# Save the figure for display
output_vars['fig'] = fig

# Let's also examine the coefficient of determination (R²)
r_squared = pearson_r ** 2
print(f"R² value: {r_squared:.2f}")
print(f"This means that {r_squared:.0%} of the variance in Variable 2 can be explained by Variable 1")

# Create a stronger correlation example for comparison
strong_x = np.random.normal(0, 1, n_samples)
strong_y = 0.9 * strong_x + np.random.normal(0, 0.1, n_samples)
strong_r, _ = stats.pearsonr(strong_x, strong_y)

# Create a weaker correlation example for comparison
weak_x = np.random.normal(0, 1, n_samples)
weak_y = 0.3 * weak_x + np.random.normal(0, 0.5, n_samples)
weak_r, _ = stats.pearsonr(weak_x, weak_y)

# Create a negative correlation example
negative_x = np.random.normal(0, 1, n_samples)
negative_y = -0.7 * negative_x + np.random.normal(0, 0.2, n_samples)
negative_r, _ = stats.pearsonr(negative_x, negative_y)

# Plot all correlations together for comparison
fig2 = go.Figure()

# Add the main correlation
fig2.add_trace(go.Scatter(
    x=x, y=y, 
    mode='markers', 
    name=f'Current (r = {pearson_r:.2f})',
    marker=dict(color='blue', size=8)
))

# Add the strong correlation
fig2.add_trace(go.Scatter(
    x=strong_x, y=strong_y, 
    mode='markers', 
    name=f'Strong (r = {strong_r:.2f})',
    marker=dict(color='green', size=8)
))

# Add the weak correlation
fig2.add_trace(go.Scatter(
    x=weak_x, y=weak_y, 
    mode='markers', 
    name=f'Weak (r = {weak_r:.2f})',
    marker=dict(color='orange', size=8)
))

# Add the negative correlation
fig2.add_trace(go.Scatter(
    x=negative_x, y=negative_y, 
    mode='markers', 
    name=f'Negative (r = {negative_r:.2f})',
    marker=dict(color='red', size=8)
))

# Update layout
fig2.update_layout(
    title='Comparison of Different Correlation Strengths',
    xaxis_title='X Variable',
    yaxis_title='Y Variable',
    height=600
)

# Save the second figure for display
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
            - Change the `correlation` value to see how it affects the scatter plot
            - Increase or decrease the `noise_level` to see how it affects the actual correlation
            - Increase `n_samples` to see how sample size affects correlation significance
            
            **Challenges:**
            1. Set the correlation to zero and observe the results
            2. Create a perfect correlation (r = 1.0) by removing noise
            3. Find the minimum sample size needed for your correlation to be statistically significant
            4. Modify the code to demonstrate correlation between variables with different scales
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
    ### Understanding Correlation Coefficients:
    
    1. **Pearson's r**: 
       - Measures linear relationship between two continuous variables
       - Ranges from -1 (perfect negative) to +1 (perfect positive)
       - Values near 0 indicate little to no linear relationship
    
    2. **Statistical Significance (p-value)**:
       - Tells us whether the correlation might be due to random chance
       - Generally, p < 0.05 is considered statistically significant
       - Larger sample sizes can make even weak correlations significant
    
    3. **Coefficient of Determination (R²)**:
       - The square of the correlation coefficient
       - Represents the proportion of variance in one variable explained by the other
       - Ranges from 0 to 1
    
    4. **Practical Interpretation**:
       - |r| < 0.3: Weak correlation
       - 0.3 ≤ |r| < 0.7: Moderate correlation
       - |r| ≥ 0.7: Strong correlation
       - These thresholds vary by field and context
    
    ### Applications in Epidemiology:
    
    Correlation analysis helps identify potential risk factors and relationships between variables such as:
    - Disease incidence and environmental factors
    - Health behaviors and outcomes
    - Socioeconomic factors and health disparities
    - Effectiveness of interventions across populations
    """)

def correlation_causation_lesson():
    st.header("Correlation vs. Causation")
    
    st.markdown("""
    ### The Distinction Between Correlation and Causation
    
    A key principle in epidemiology is that **correlation does not imply causation**. 
    
    Just because two variables are correlated does not mean one causes the other. The relationship could be due to:
    
    1. **Coincidence**: The correlation occurred by chance
    2. **Common cause**: A third variable causes both
    3. **Reverse causality**: The presumed effect actually causes the presumed cause
    4. **Confounding**: Other variables influence the relationship
    
    Let's explore this concept with some examples:
    """)
    
    # Initial code example
    initial_code = """# Exploring correlation vs. causation
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 100    # Number of data points
alpha = 0.75       # Influence of common cause (try 0 to 1)
direct_effect = 0  # Direct causal effect (try 0 for pure confounding)

# Generate data with a common cause (confounder)
np.random.seed(42)  # For reproducible results

# Common cause variable (e.g., socioeconomic status)
common_cause = np.random.normal(0, 1, n_samples)

# Two variables both influenced by the common cause
# (e.g., education level and health outcomes)
variable_a = alpha * common_cause + np.random.normal(0, 0.5, n_samples)
variable_b = alpha * common_cause + direct_effect * variable_a + np.random.normal(0, 0.5, n_samples)

# Create a DataFrame with the data
data = pd.DataFrame({
    'Common Cause': common_cause,
    'Variable A': variable_a,
    'Variable B': variable_b
})

# Calculate correlations
corr_ab = stats.pearsonr(variable_a, variable_b)[0]
corr_ac = stats.pearsonr(variable_a, common_cause)[0]
corr_bc = stats.pearsonr(variable_b, common_cause)[0]

# Print results
print(f"Correlation between Variable A and Variable B: {corr_ab:.2f}")
print(f"Correlation between Variable A and Common Cause: {corr_ac:.2f}")
print(f"Correlation between Variable B and Common Cause: {corr_bc:.2f}")

# Create scatter plot for A vs B
fig = px.scatter(
    data, 
    x='Variable A', 
    y='Variable B',
    title=f"Variable A vs Variable B (r = {corr_ab:.2f})",
    trendline='ols'
)
output_vars['fig'] = fig

# Create visualizations with all three variables
fig2 = go.Figure()

# A vs B - Colored by Common Cause
fig2 = px.scatter(
    data, 
    x='Variable A', 
    y='Variable B',
    color='Common Cause',
    title="A vs B Colored by Common Cause",
    color_continuous_scale='RdBu_r'
)
output_vars['fig2'] = fig2

# Partial correlation - controlling for the common cause
# Calculate residuals after regressing out the common cause
model_a = stats.linregress(common_cause, variable_a)
model_b = stats.linregress(common_cause, variable_b)

residual_a = variable_a - (model_a.slope * common_cause + model_a.intercept)
residual_b = variable_b - (model_b.slope * common_cause + model_b.intercept)

# Calculate partial correlation
partial_corr = stats.pearsonr(residual_a, residual_b)[0]

print(f"\\nPartial correlation between A and B (controlling for Common Cause): {partial_corr:.2f}")

if abs(partial_corr) < abs(corr_ab):
    print(f"The partial correlation is weaker than the direct correlation ({abs(partial_corr):.2f} vs {abs(corr_ab):.2f})")
    print("This suggests the Common Cause explains some of the relationship between A and B")
    
    percent_reduction = (1 - abs(partial_corr) / abs(corr_ab)) * 100
    print(f"The correlation is reduced by {percent_reduction:.1f}% when controlling for the Common Cause")
else:
    print("The partial correlation is not weaker, suggesting the Common Cause is not a confounder")

# Plot the residuals (partial correlation visualization)
residual_data = pd.DataFrame({
    'Residual A': residual_a,
    'Residual B': residual_b
})

fig3 = px.scatter(
    residual_data,
    x='Residual A',
    y='Residual B',
    title=f"Partial Correlation (r = {partial_corr:.2f})",
    trendline='ols'
)
output_vars['fig3'] = fig3

# Create a diagram illustrating the relationship
fig4 = go.Figure()

# Add nodes (circles)
fig4.add_trace(go.Scatter(
    x=[0, -1, 1], 
    y=[1, 0, 0],
    mode='markers+text',
    marker=dict(size=[20, 15, 15], color=['red', 'blue', 'blue']),
    text=['Common<br>Cause', 'Variable A', 'Variable B'],
    textposition="bottom center",
    name='Variables'
))

# Add arrows
# Common Cause to A
fig4.add_shape(
    type="line", x0=-0.15, y0=0.85, x1=-0.85, y1=0.15,
    line=dict(color="black", width=2, dash="solid"),
)
# Add arrowhead
fig4.add_annotation(
    x=-0.85, y=0.15, 
    text="",
    showarrow=True,
    axref="x", ayref="y",
    ax=-0.15, ay=0.85,
    arrowhead=3,
    arrowwidth=2,
    arrowcolor="black",
)

# Common Cause to B
fig4.add_shape(
    type="line", x0=0.15, y0=0.85, x1=0.85, y1=0.15,
    line=dict(color="black", width=2, dash="solid"),
)
# Add arrowhead
fig4.add_annotation(
    x=0.85, y=0.15, 
    text="",
    showarrow=True,
    axref="x", ayref="y",
    ax=0.15, ay=0.85,
    arrowhead=3,
    arrowwidth=2,
    arrowcolor="black",
)

# A to B (dashed line for the correlation, not causation)
fig4.add_shape(
    type="line", x0=-0.85, y0=0, x1=0.85, y1=0,
    line=dict(color="black", width=2, dash="dash"),
)

# Direct effect if any
if direct_effect > 0:
    fig4.add_annotation(
        x=0, y=0, 
        text="",
        showarrow=True,
        axref="x", ayref="y",
        ax=-0.85, ay=0,
        arrowhead=3,
        arrowwidth=2,
        arrowcolor="green",
    )
    fig4.add_annotation(
        x=0, y=-0.2,
        text=f"Direct Effect: {direct_effect:.2f}",
        showarrow=False,
        font=dict(color="green")
    )

# Add correlation values
fig4.add_annotation(
    x=-0.5, y=0.5,
    text=f"r = {corr_ac:.2f}",
    showarrow=False,
    font=dict(color="black")
)

fig4.add_annotation(
    x=0.5, y=0.5,
    text=f"r = {corr_bc:.2f}",
    showarrow=False,
    font=dict(color="black")
)

fig4.add_annotation(
    x=0, y=0.1,
    text=f"r = {corr_ab:.2f}",
    showarrow=False,
    font=dict(color="black")
)

# Update layout
fig4.update_layout(
    title='Causal Diagram: Common Cause Creates Correlation',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
    showlegend=False,
    height=400
)
output_vars['fig4'] = fig4
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
            - Set `alpha` to 0 (no common cause) and observe the correlation
            - Set `alpha` to 1 (strong common cause) to see maximum confounding
            - Add `direct_effect` (e.g., 0.5) to include a true causal relationship
            - Change `n_samples` to see how sample size affects the results
            
            **Challenges:**
            1. Create a scenario where the correlation is entirely due to confounding
            2. Create a scenario with both confounding and direct causation
            3. Find a combination where the partial correlation is close to zero
            4. Simulate a case of reverse causality (hint: make B influence A instead)
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
                
            if 'fig4' in output_vars:
                st.plotly_chart(output_vars['fig4'], use_container_width=True)
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Concepts")
    st.markdown("""
    ### Types of Spurious Correlations:
    
    1. **Confounding**:
       - A third variable influences both the exposure and outcome
       - Example: Ice cream sales and drowning deaths (both influenced by hot weather)
    
    2. **Selection Bias**:
       - The way subjects are selected creates an artificial correlation
       - Example: Hospital studies showing mild cases recovering worse than severe cases
    
    3. **Information Bias**:
       - Systematic errors in how data is collected create false correlations
       - Example: Self-reported data that varies by demographic factors
    
    4. **Ecological Fallacy**:
       - Inferring individual relationships from group-level data
       - Example: Countries with more doctors having higher disease rates
    
    ### Methods to Establish Causation:
    
    1. **Bradford Hill Criteria**:
       - Strength of association
       - Consistency
       - Specificity
       - Temporality (cause precedes effect)
       - Biological gradient (dose-response)
       - Plausibility
       - Coherence with existing knowledge
       - Experimental evidence
       - Analogy to established causes
    
    2. **Study Designs for Causation**:
       - Randomized controlled trials
       - Cohort studies
       - Case-control studies
       - Natural experiments
    
    3. **Statistical Approaches**:
       - Adjustment for confounders
       - Propensity score matching
       - Instrumental variables
       - Causal inference methods
    """)

def correlation_types_lesson():
    st.header("Types of Correlation")
    
    st.markdown("""
    ### Different Measures of Correlation
    
    There are several ways to measure correlation, each with specific uses:
    
    1. **Pearson Correlation (r)**: Measures linear relationships between continuous variables
    2. **Spearman Correlation (ρ)**: Measures monotonic relationships (doesn't have to be linear)
    3. **Kendall's Tau (τ)**: Another rank correlation; more robust to outliers than Spearman
    4. **Point-Biserial Correlation**: Between continuous and binary variables
    
    Let's compare these different correlation types:
    """)
    
    # Initial code example
    initial_code = """# Comparing different correlation measures
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 100          # Number of data points
outlier_strength = 5     # How extreme the outliers are (try 10 or 20)
add_outliers = True      # Whether to include outliers
use_monotonic = True     # Whether to use monotonically related data

# Generate data for different scenarios
np.random.seed(42)  # For reproducible results

# 1. Linear relationship with noise
x_linear = np.linspace(-3, 3, n_samples)
y_linear = 0.7 * x_linear + np.random.normal(0, 1, n_samples)

# 2. Linear relationship with outliers
x_outliers = np.copy(x_linear)
y_outliers = np.copy(y_linear)

# Add some outliers
if add_outliers:
    outlier_indices = [5, 30, 70, 95]
    for idx in outlier_indices:
        y_outliers[idx] = outlier_strength * np.sign(np.random.randn())  # Random extreme value

# 3. Monotonic but non-linear relationship
if use_monotonic:
    x_monotonic = np.linspace(-3, 3, n_samples)
    y_monotonic = np.exp(x_monotonic) + np.random.normal(0, 2, n_samples)
else:
    # Use linear data as fallback
    x_monotonic = np.copy(x_linear)
    y_monotonic = np.copy(y_linear)

# Create DataFrames
df_linear = pd.DataFrame({'X': x_linear, 'Y': y_linear, 'Type': 'Linear'})
df_outliers = pd.DataFrame({'X': x_outliers, 'Y': y_outliers, 'Type': 'With Outliers'})
df_monotonic = pd.DataFrame({'X': x_monotonic, 'Y': y_monotonic, 'Type': 'Monotonic'})

# Combine data for visualization
all_data = pd.concat([df_linear, df_outliers, df_monotonic])

# Calculate different correlation types
correlations = []

# Linear data
pearson_linear = stats.pearsonr(x_linear, y_linear)[0]
spearman_linear = stats.spearmanr(x_linear, y_linear)[0]
kendall_linear = stats.kendalltau(x_linear, y_linear)[0]

correlations.append({
    'Dataset': 'Linear',
    'Pearson': pearson_linear,
    'Spearman': spearman_linear,
    'Kendall': kendall_linear
})

# Data with outliers
pearson_outliers = stats.pearsonr(x_outliers, y_outliers)[0]
spearman_outliers = stats.spearmanr(x_outliers, y_outliers)[0]
kendall_outliers = stats.kendalltau(x_outliers, y_outliers)[0]

correlations.append({
    'Dataset': 'With Outliers',
    'Pearson': pearson_outliers,
    'Spearman': spearman_outliers,
    'Kendall': kendall_outliers
})

# Monotonic data
pearson_monotonic = stats.pearsonr(x_monotonic, y_monotonic)[0]
spearman_monotonic = stats.spearmanr(x_monotonic, y_monotonic)[0]
kendall_monotonic = stats.kendalltau(x_monotonic, y_monotonic)[0]

correlations.append({
    'Dataset': 'Monotonic',
    'Pearson': pearson_monotonic,
    'Spearman': spearman_monotonic,
    'Kendall': kendall_monotonic
})

# Create a DataFrame of correlations
corr_df = pd.DataFrame(correlations)

# Print the correlation summary
print("Correlation Coefficients by Method and Dataset:")
print(corr_df.round(3))

# Plot the data
fig = px.scatter(
    all_data, 
    x='X', 
    y='Y', 
    color='Type',
    facet_col='Type',
    title='Comparison of Different Data Relationships',
    trendline='ols'
)

# Update layout for better separation
fig.update_layout(height=600)
output_vars['fig'] = fig

# Create a bar chart of correlations
corr_long = pd.melt(
    corr_df, 
    id_vars=['Dataset'], 
    value_vars=['Pearson', 'Spearman', 'Kendall'],
    var_name='Method',
    value_name='Correlation'
)

fig2 = px.bar(
    corr_long,
    x='Dataset',
    y='Correlation',
    color='Method',
    barmode='group',
    title='Correlation Coefficients by Method and Dataset',
    height=500
)

output_vars['fig2'] = fig2

# Linear vs Monotonic visualization
if use_monotonic:
    # Create a side-by-side comparison of linear and monotonic
    compare_df = pd.DataFrame({
        'X': np.concatenate([x_linear, x_monotonic]),
        'Y': np.concatenate([y_linear, y_monotonic]),
        'Type': ['Linear'] * n_samples + ['Monotonic'] * n_samples
    })
    
    # Create scatter plots with OLS and LOWESS trendlines
    fig3 = px.scatter(
        compare_df,
        x='X',
        y='Y',
        color='Type',
        facet_col='Type',
        trendline='ols',
        title='Linear vs. Monotonic Relationships'
    )
    
    output_vars['fig3'] = fig3
    
    # Print explanation
    print("\\nUnderstanding Linear vs Monotonic Relationships:")
    print(f"- For Linear data: Pearson={pearson_linear:.3f}, Spearman={spearman_linear:.3f}")
    print(f"- For Monotonic data: Pearson={pearson_monotonic:.3f}, Spearman={spearman_monotonic:.3f}")
    
    if abs(spearman_monotonic) > abs(pearson_monotonic):
        print("Notice that Spearman correlation is higher than Pearson for the monotonic relationship.")
        print("This demonstrates that Spearman better captures non-linear but monotonic relationships.")
    
    # Explain when to use which correlation
    print("\\nWhen to use each correlation type:")
    print("- Pearson: When you expect a linear relationship and have normally distributed data")
    print("- Spearman: When you're interested in monotonic relationships or have ordinal data")
    print("- Kendall: Similar to Spearman but better for small samples and more robust to outliers")
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
            - Set `add_outliers = False` to see correlation without outliers
            - Increase `outlier_strength` to see how extreme outliers affect correlations
            - Toggle `use_monotonic = False` to compare with linear data
            - Increase `n_samples` to see the effect on stability of the correlations
            
            **Challenges:**
            1. Create a dataset where Pearson and Spearman correlations have opposite signs
            2. Find the minimum number of outliers needed to reduce Pearson correlation below 0.5
            3. Create a scenario where Kendall's Tau is more appropriate than Spearman
            4. Generate a U-shaped relationship and analyze the correlations
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
    ### When to Use Each Correlation Type:
    
    1. **Pearson Correlation (r)**:
       - For linear relationships between continuous variables
       - Both variables should be normally distributed
       - Sensitive to outliers
       - Example: BMI and blood pressure
    
    2. **Spearman Correlation (ρ)**:
       - For monotonic relationships (consistently increasing or decreasing)
       - Works with ordinal, interval, or ratio data
       - Less sensitive to outliers (uses ranks)
       - Example: Disease severity ranking and quality of life score
    
    3. **Kendall's Tau (τ)**:
       - Similar to Spearman but uses concordant and discordant pairs
       - More robust with small sample sizes
       - More robust to outliers than Spearman
       - Example: Comparing two clinicians' rankings of patient condition
    
    4. **Point-Biserial Correlation**:
       - Between a continuous variable and a binary variable
       - Special case of Pearson correlation
       - Example: Relationship between treatment (yes/no) and blood pressure
    
    ### Common Pitfalls in Correlation Analysis:
    
    1. **Outliers**: Can dramatically affect Pearson correlation but have less impact on rank correlations
        
    2. **Restricted Range**: Correlation can appear weaker if the range of values is restricted
    
    3. **Small Sample Size**: Correlations from small samples are less reliable and have wider confidence intervals
    
    4. **Ecological Correlations**: Group-level correlations may differ from individual-level correlations
    """)

def public_health_examples_lesson():
    st.header("Public Health Correlation Examples")
    
    st.markdown("""
    ### Correlation Analysis in Public Health and Epidemiology
    
    Correlation analysis is widely used in public health to:
    
    - Identify potential risk factors
    - Study social determinants of health
    - Evaluate intervention effects
    - Understand disease patterns
    
    Let's explore some realistic public health examples:
    """)
    
    # Initial code example
    initial_code = """# Analyzing public health correlations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_counties = 100          # Number of geographic units (e.g., counties)
confounding = True        # Include socioeconomic confounding
sample_dataset = "vaccination"  # Options: "vaccination", "pollution", "social_determinants", "disease_outbreak"

# Generate simulated data
np.random.seed(42)  # For reproducible results

if sample_dataset == "vaccination":
    # Dataset: Vaccination rates and disease incidence
    
    # Step 1: Generate county-level data
    # Socioeconomic status (confounder)
    socioeconomic_status = np.random.normal(0, 1, n_counties)
    
    # Vaccination rate influenced by SES if confounding is enabled
    if confounding:
        vaccination_rate = 70 + 15 * socioeconomic_status + np.random.normal(0, 5, n_counties)
    else:
        vaccination_rate = 70 + np.random.normal(0, 10, n_counties)
    
    # Clip vaccination rates to realistic range
    vaccination_rate = np.clip(vaccination_rate, 30, 95)
    
    # Disease incidence rate - influenced by vaccination and SES
    disease_incidence = 100 - 0.8 * vaccination_rate
    if confounding:
        disease_incidence = disease_incidence - 5 * socioeconomic_status
    disease_incidence = disease_incidence + np.random.normal(0, 10, n_counties)
    disease_incidence = np.clip(disease_incidence, 0, 100)
    
    # Create DataFrame
    counties = [f"County {i+1}" for i in range(n_counties)]
    data = pd.DataFrame({
        'County': counties,
        'Vaccination Rate (%)': vaccination_rate,
        'Disease Incidence (per 100k)': disease_incidence,
        'Socioeconomic Status': socioeconomic_status
    })
    
    # Set titles
    main_title = "Relationship Between Vaccination Rates and Disease Incidence"
    x_title = "Vaccination Rate (%)"
    y_title = "Disease Incidence (per 100,000 population)"
    
elif sample_dataset == "pollution":
    # Dataset: Air pollution and respiratory disease
    
    # Generate county-level data
    population_density = np.random.exponential(1, n_counties)
    
    # PM2.5 levels influenced by population density if confounding is enabled
    if confounding:
        pm25_levels = 8 + 5 * population_density + np.random.normal(0, 2, n_counties)
    else:
        pm25_levels = 12 + np.random.normal(0, 4, n_counties)
    
    # Clip PM2.5 to realistic range
    pm25_levels = np.clip(pm25_levels, 2, 35)
    
    # Asthma rate - influenced by PM2.5 and population density (as proxy for healthcare access)
    asthma_rate = 5 + 0.8 * pm25_levels
    if confounding:
        asthma_rate = asthma_rate + 2 * population_density
    asthma_rate = asthma_rate + np.random.normal(0, 2, n_counties)
    asthma_rate = np.clip(asthma_rate, 2, 30)
    
    # Create DataFrame
    counties = [f"County {i+1}" for i in range(n_counties)]
    data = pd.DataFrame({
        'County': counties,
        'PM2.5 Levels (μg/m³)': pm25_levels,
        'Asthma Rate (%)': asthma_rate,
        'Population Density': population_density
    })
    
    # Set titles
    main_title = "Relationship Between Air Pollution and Asthma Rates"
    x_title = "PM2.5 Levels (μg/m³)"
    y_title = "Asthma Rate (%)"
    
elif sample_dataset == "social_determinants":
    # Dataset: Education, income, and life expectancy
    
    # Generate county-level data
    education_years = np.random.normal(13, 2, n_counties)
    
    # Income influenced by education
    household_income = 20000 + 5000 * (education_years - 10)
    household_income = household_income + np.random.normal(0, 10000, n_counties)
    household_income = np.clip(household_income, 15000, 150000)
    
    # Life expectancy - influenced by both education and income
    life_expectancy = 70 + 0.5 * education_years
    if confounding:
        life_expectancy = life_expectancy + household_income / 50000
    life_expectancy = life_expectancy + np.random.normal(0, 2, n_counties)
    life_expectancy = np.clip(life_expectancy, 65, 90)
    
    # Create DataFrame
    counties = [f"County {i+1}" for i in range(n_counties)]
    data = pd.DataFrame({
        'County': counties,
        'Education (Years)': education_years,
        'Household Income ($)': household_income,
        'Life Expectancy (Years)': life_expectancy
    })
    
    # Set titles
    main_title = "Relationship Between Education and Life Expectancy"
    x_title = "Education (Years)"
    y_title = "Life Expectancy (Years)"
    
else:  # disease_outbreak
    # Dataset: Disease cases over time, demonstrating autocorrelation
    
    # Generate weekly disease cases with seasonal pattern and trend
    weeks = 52  # One year of data
    
    # Time index
    time = np.arange(weeks)
    
    # Baseline trend (increasing)
    baseline = 50 + time * 0.5
    
    # Seasonal component
    seasonal = 20 * np.sin(2 * np.pi * time / 52)
    
    # Generate cases
    cases = baseline + seasonal + np.random.normal(0, 10, weeks)
    cases = np.clip(cases, 0, None)
    
    # Population size (constant or growing)
    if confounding:
        # Growing population affects both cases and rates
        population = 100000 * (1 + 0.005 * time)
    else:
        # Constant population
        population = 100000 * np.ones(weeks)
    
    # Calculate rates
    rates = cases / (population / 100000)
    
    # Create lagged variables (previous week's cases)
    lagged_cases = np.concatenate([[np.nan], cases[:-1]])
    
    # Create DataFrame
    data = pd.DataFrame({
        'Week': time + 1,
        'Cases': cases,
        'Population': population,
        'Rate (per 100k)': rates,
        'Previous Week Cases': lagged_cases
    })
    
    # Set titles
    main_title = "Disease Cases Over Time"
    x_title = "Week"
    y_title = "Number of Cases"
    
    # For this dataset, we'll create a different visualization
    # Create time series plot
    fig = px.line(
        data,
        x='Week',
        y=['Cases', 'Rate (per 100k)'],
        title=main_title,
        labels={'value': 'Value', 'variable': 'Metric'}
    )
    
    # Calculate autocorrelation
    autocorr = data['Cases'].autocorr()
    
    # Create autocorrelation plot
    lag_data = data[['Cases', 'Previous Week Cases']].dropna()
    
    fig2 = px.scatter(
        lag_data,
        x='Previous Week Cases',
        y='Cases',
        title=f'Autocorrelation of Cases (r = {autocorr:.2f})',
        trendline='ols'
    )
    
    # Store special figures for this dataset
    output_vars['fig_special'] = fig
    output_vars['fig2_special'] = fig2
    
    # Calculate trend
    data['Time'] = data['Week']
    trend_model = stats.linregress(data['Time'], data['Cases'])
    trend_correlation = trend_model.rvalue
    
    # Print special results for time series
    print(f"Dataset: Disease outbreak time series")
    print(f"Autocorrelation (lag 1): {autocorr:.2f}")
    print(f"Trend correlation: {trend_correlation:.2f}")
    print(f"Trend p-value: {trend_model.pvalue:.4f}")
    
    # Return from this dataset to avoid standard analysis
    output_vars['dataset_type'] = 'time_series'
    
    # For the autocorrelation plot
    fig3 = go.Figure()
    
    # Calculate autocorrelation for different lags
    max_lag = 10
    autocorr_values = []
    for lag in range(1, max_lag + 1):
        lagged = data['Cases'].shift(lag)
        valid = ~np.isnan(lagged)
        if sum(valid) > 0:
            ac = np.corrcoef(data['Cases'][valid], lagged[valid])[0, 1]
            autocorr_values.append(ac)
        else:
            autocorr_values.append(0)
    
    # Plot autocorrelation function
    fig3.add_trace(go.Bar(
        x=list(range(1, max_lag + 1)),
        y=autocorr_values,
        name='Autocorrelation'
    ))
    
    # Add significance lines (approximate 95% CI)
    sig_level = 1.96 / np.sqrt(len(data))
    fig3.add_shape(
        type="line",
        x0=0.5, x1=max_lag + 0.5, y0=sig_level, y1=sig_level,
        line=dict(color="red", dash="dash")
    )
    
    fig3.add_shape(
        type="line",
        x0=0.5, x1=max_lag + 0.5, y0=-sig_level, y1=-sig_level,
        line=dict(color="red", dash="dash")
    )
    
    fig3.update_layout(
        title='Autocorrelation Function (ACF)',
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        yaxis_range=[-1, 1]
    )
    
    output_vars['fig3_special'] = fig3
    
    # Exit this dataset option
    if 'dataset_type' in output_vars and output_vars['dataset_type'] == 'time_series':
        print("\\nNote: Time series data requires special correlation analysis methods.")
        print("Conventional Pearson correlation may not be appropriate due to:")
        print("- Autocorrelation between adjacent time points")
        print("- Potential non-stationarity (trend, seasonality)")
        print("- Time-dependent variance")
        
        print("\\nMore appropriate methods include:")
        print("- Autocorrelation and partial autocorrelation functions")
        print("- Correlation between detrended/deseasonalized series")
        print("- Cross-correlation functions for comparing time series")

# Skip standard analysis for time series data
if 'dataset_type' in output_vars and output_vars['dataset_type'] == 'time_series':
    # We've already created custom visualizations for time series
    pass
else:
    # Perform analysis for standard datasets
    
    # Determine variables for correlation
    if sample_dataset == "vaccination":
        x_var = 'Vaccination Rate (%)'
        y_var = 'Disease Incidence (per 100k)'
        confound_var = 'Socioeconomic Status'
    elif sample_dataset == "pollution":
        x_var = 'PM2.5 Levels (μg/m³)'
        y_var = 'Asthma Rate (%)'
        confound_var = 'Population Density'
    elif sample_dataset == "social_determinants":
        x_var = 'Education (Years)'
        y_var = 'Life Expectancy (Years)'
        confound_var = 'Household Income ($)'
    
    # Calculate Pearson correlation
    pearson_r, p_value = stats.pearsonr(data[x_var], data[y_var])
    
    # Print results
    print(f"Dataset: {sample_dataset}")
    print(f"Correlation between {x_var} and {y_var}: r = {pearson_r:.3f}, p = {p_value:.4f}")
    
    if confounding:
        print(f"Confounding by {confound_var} is present in the data")
        
        # Calculate partial correlation (controlling for confounder)
        x_resid = stats.linregress(data[confound_var], data[x_var]).intercept + \
                 stats.linregress(data[confound_var], data[x_var]).resid
        y_resid = stats.linregress(data[confound_var], data[y_var]).intercept + \
                 stats.linregress(data[confound_var], data[y_var]).resid
        
        partial_r, partial_p = stats.pearsonr(x_resid, y_resid)
        
        print(f"Partial correlation (controlling for {confound_var}): r = {partial_r:.3f}, p = {partial_p:.4f}")
        
        # Check if confounding changed the correlation
        if abs(partial_r - pearson_r) > 0.1:
            print(f"Substantial confounding effect detected.")
            percent_change = (partial_r - pearson_r) / pearson_r * 100
            if abs(percent_change) > 25:
                print(f"The correlation changed by {abs(percent_change):.1f}% when controlling for {confound_var}")
    
    # Create scatter plot
    fig = px.scatter(
        data, 
        x=x_var, 
        y=y_var,
        color=confound_var if confounding else None,
        title=main_title,
        trendline='ols',
        labels={x_var: x_title, y_var: y_title}
    )
    
    # Adjust layout
    fig.update_layout(height=600)
    output_vars['fig'] = fig
    
    # If confounding is present, create a partial correlation visualization
    if confounding:
        # Create a figure for the partial correlation
        fig2 = go.Figure()
        
        # Add scatter plot for residuals
        fig2.add_trace(go.Scatter(
            x=x_resid,
            y=y_resid,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue')
        ))
        
        # Add trend line
        slope, intercept = np.polyfit(x_resid, y_resid, 1)
        x_range = np.linspace(min(x_resid), max(x_resid), 100)
        y_pred = intercept + slope * x_range
        
        fig2.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name='Trend Line',
            line=dict(color='red')
        ))
        
        # Update layout
        fig2.update_layout(
            title=f"Partial Correlation (controlling for {confound_var}): r = {partial_r:.3f}",
            xaxis_title=f"Residuals of {x_var}",
            yaxis_title=f"Residuals of {y_var}",
            height=500
        )
        
        output_vars['fig2'] = fig2
        
        # Create a figure showing the confounding structure
        fig3 = go.Figure()
        
        # Add nodes for variables
        fig3.add_trace(go.Scatter(
            x=[0, -1, 1], 
            y=[1, 0, 0],
            mode='markers+text',
            marker=dict(size=[20, 15, 15], color=['red', 'blue', 'blue']),
            text=[confound_var, x_var, y_var],
            textposition="bottom center",
            name='Variables'
        ))
        
        # Add arrows for relationships
        
        # Confounder to X
        x_conf_corr = stats.pearsonr(data[confound_var], data[x_var])[0]
        line_width = abs(x_conf_corr) * 5  # Scale line width by correlation strength
        
        fig3.add_shape(
            type="line", x0=-0.15, y0=0.85, x1=-0.85, y1=0.15,
            line=dict(color="black", width=line_width, dash="solid"),
        )
        # Add arrowhead
        fig3.add_annotation(
            x=-0.85, y=0.15, 
            text="",
            showarrow=True,
            axref="x", ayref="y",
            ax=-0.15, ay=0.85,
            arrowhead=3,
            arrowwidth=line_width,
            arrowcolor="black",
        )
        
        # Confounder to Y
        y_conf_corr = stats.pearsonr(data[confound_var], data[y_var])[0]
        line_width = abs(y_conf_corr) * 5  # Scale line width by correlation strength
        
        fig3.add_shape(
            type="line", x0=0.15, y0=0.85, x1=0.85, y1=0.15,
            line=dict(color="black", width=line_width, dash="solid"),
        )
        # Add arrowhead
        fig3.add_annotation(
            x=0.85, y=0.15, 
            text="",
            showarrow=True,
            axref="x", ayref="y",
            ax=0.15, ay=0.85,
            arrowhead=3,
            arrowwidth=line_width,
            arrowcolor="black",
        )
        
        # X to Y (dashed line for the correlation, not necessarily causation)
        line_width = abs(pearson_r) * 5  # Scale line width by correlation strength
        
        fig3.add_shape(
            type="line", x0=-0.85, y0=0, x1=0.85, y1=0,
            line=dict(color="black", width=line_width, dash="dash"),
        )
        
        # Add correlation values as annotations
        fig3.add_annotation(
            x=-0.5, y=0.5,
            text=f"r = {x_conf_corr:.2f}",
            showarrow=False,
            font=dict(color="black")
        )
        
        fig3.add_annotation(
            x=0.5, y=0.5,
            text=f"r = {y_conf_corr:.2f}",
            showarrow=False,
            font=dict(color="black")
        )
        
        fig3.add_annotation(
            x=0, y=0.1,
            text=f"r = {pearson_r:.2f}\nPartial r = {partial_r:.2f}",
            showarrow=False,
            font=dict(color="black")
        )
        
        # Update layout
        fig3.update_layout(
            title='Confounding Structure',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
            showlegend=False,
            height=400
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
        # Add hints and challenges
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Change `sample_dataset` to explore different public health scenarios
            - Toggle `confounding` to see its effect on correlations
            - Adjust `n_counties` to see how sample size affects results
            
            **Challenges:**
            1. Modify the vaccination dataset to create herd immunity (non-linear relationship)
            2. Create a dataset where confounding reverses the direction of correlation
            3. Add a third confounding variable to the social determinants dataset
            4. Create a seasonal pattern for the disease outbreak dataset
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
            if 'dataset_type' in output_vars and output_vars['dataset_type'] == 'time_series':
                # Special time series figures
                if 'fig_special' in output_vars:
                    st.plotly_chart(output_vars['fig_special'], use_container_width=True)
                if 'fig2_special' in output_vars:
                    st.plotly_chart(output_vars['fig2_special'], use_container_width=True)
                if 'fig3_special' in output_vars:
                    st.plotly_chart(output_vars['fig3_special'], use_container_width=True)
            else:
                # Standard figures
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
    ### Applications of Correlation in Public Health:
    
    1. **Disease Surveillance**:
       - Identifying geographic patterns in disease occurrence
       - Detecting outbreaks through temporal correlations
       - Monitoring effectiveness of interventions
    
    2. **Risk Factor Analysis**:
       - Assessing relationships between exposures and outcomes
       - Screening potential risk factors for further study
       - Quantifying strength of associations
    
    3. **Social Determinants of Health**:
       - Understanding how social factors correlate with health outcomes
       - Measuring health disparities across populations
       - Identifying areas for policy interventions
    
    4. **Health Systems Analysis**:
       - Correlating healthcare access with outcomes
       - Assessing healthcare resource distribution
       - Evaluating impact of public health programs
    
    ### Methodological Considerations:
    
    1. **Ecological Fallacy**:
       - County-level correlations may not reflect individual relationships
       - Avoid inferring individual risks from population-level data
    
    2. **Confounding in Public Health**:
       - Socioeconomic status often confounds many health relationships
       - Demographic factors can create spurious correlations
       - Always consider plausible confounders
    
    3. **Appropriate Correlation Measures**:
       - Count data often requires special approaches
       - Rates vs. absolute numbers can yield different results
       - Time-series data needs special correlation methods
    
    4. **From Correlation to Intervention**:
       - Correlations help identify potential targets
       - Causal inference requires additional evidence
       - Intervention studies provide stronger evidence of causality
    """)

if __name__ == "__main__":
    app()