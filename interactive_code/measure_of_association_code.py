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
    st.title("Interactive Coding Laboratory: Measures of Association")
    
    st.markdown("""
    ## Learn by Coding: Measures of Association
    
    This interactive coding laboratory allows you to explore and understand different measures 
    of association used in epidemiology. Modify the example code and see how changes affect the results.
    
    Choose a topic to explore:
    """)
    
    # Simple topic selection for first-year students - just 2 options
    topic = st.selectbox(
        "Select a topic:",
        ["Basic Measures Calculation", 
         "Comparing Measures of Association"]
    )
    
    # Display the selected topic
    if topic == "Basic Measures Calculation":
        basic_measures_lesson()
    elif topic == "Comparing Measures of Association":
        comparing_measures_lesson()

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

def basic_measures_lesson():
    st.header("Basic Measures of Association Calculation")
    
    st.markdown("""
    ### Understanding Measures of Association
    
    Measures of association quantify the relationship between an exposure and an outcome.
    They help us understand how strongly an exposure is related to a disease or health outcome.
    
    Let's calculate the three most common measures:
    - **Risk Ratio (RR)** - Also known as relative risk
    - **Odds Ratio (OR)**
    - **Risk Difference (RD)** - Also known as attributable risk
    """)
    
    # Initial code example - simplified for beginners
    initial_code = """# Basic calculation of measures of association
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
# Data for a 2x2 contingency table
exposed_cases = 40        # People who were exposed and got the disease
exposed_noncases = 60     # People who were exposed but didn't get the disease
unexposed_cases = 20      # People who weren't exposed but got the disease
unexposed_noncases = 80   # People who weren't exposed and didn't get the disease

# Create a data frame for visualization
data = pd.DataFrame({
    'Exposure': ['Exposed', 'Exposed', 'Unexposed', 'Unexposed'],
    'Outcome': ['Case', 'Non-case', 'Case', 'Non-case'],
    'Count': [exposed_cases, exposed_noncases, unexposed_cases, unexposed_noncases]
})

# Display the 2x2 table
print("2x2 Contingency Table:")
contingency_table = pd.DataFrame({
    'Cases': [exposed_cases, unexposed_cases],
    'Non-cases': [exposed_noncases, unexposed_noncases]
}, index=['Exposed', 'Unexposed'])
print(contingency_table)

# Calculate risks
total_exposed = exposed_cases + exposed_noncases
total_unexposed = unexposed_cases + unexposed_noncases

risk_exposed = exposed_cases / total_exposed
risk_unexposed = unexposed_cases / total_unexposed

print(f"\\nRisk in exposed group: {risk_exposed:.3f} or {risk_exposed*100:.1f}%")
print(f"Risk in unexposed group: {risk_unexposed:.3f} or {risk_unexposed*100:.1f}%")

# Calculate odds
odds_exposed = exposed_cases / exposed_noncases
odds_unexposed = unexposed_cases / unexposed_noncases

print(f"\\nOdds in exposed group: {odds_exposed:.3f}")
print(f"Odds in unexposed group: {odds_unexposed:.3f}")

# Calculate measures of association
# 1. Risk Ratio (RR)
risk_ratio = risk_exposed / risk_unexposed
print(f"\\nRisk Ratio (RR): {risk_ratio:.2f}")
if risk_ratio > 1:
    print(f"Interpretation: Exposure is associated with {risk_ratio:.1f}x increased risk of disease")
elif risk_ratio < 1:
    print(f"Interpretation: Exposure is associated with {1/risk_ratio:.1f}x decreased risk of disease")
else:
    print("Interpretation: No association between exposure and disease")

# 2. Odds Ratio (OR)
odds_ratio = odds_exposed / odds_unexposed
print(f"\\nOdds Ratio (OR): {odds_ratio:.2f}")
if odds_ratio > 1:
    print(f"Interpretation: The odds of disease are {odds_ratio:.1f}x higher in the exposed group")
elif odds_ratio < 1:
    print(f"Interpretation: The odds of disease are {1/odds_ratio:.1f}x lower in the exposed group")
else:
    print("Interpretation: No association between exposure and disease")

# 3. Risk Difference (RD)
risk_difference = risk_exposed - risk_unexposed
print(f"\\nRisk Difference (RD): {risk_difference:.3f} or {risk_difference*100:.1f}%")
if risk_difference > 0:
    print(f"Interpretation: Exposure is associated with {risk_difference*100:.1f}% increased absolute risk")
elif risk_difference < 0:
    print(f"Interpretation: Exposure is associated with {-risk_difference*100:.1f}% decreased absolute risk")
else:
    print("Interpretation: No difference in risk between exposed and unexposed")

# Calculate Number Needed to Harm/Treat
if risk_difference != 0:
    nnth = abs(1 / risk_difference)
    print(f"\\nNumber Needed to Harm/Treat: {nnth:.1f}")
    if risk_difference > 0:
        print(f"Interpretation: Need to expose {nnth:.1f} people to cause one additional case")
    else:
        print(f"Interpretation: Need to expose {nnth:.1f} people to prevent one case")

# Create a grouped bar chart
fig = px.bar(
    data,
    x='Exposure',
    y='Count',
    color='Outcome',
    barmode='group',
    title='Distribution of Cases and Non-cases by Exposure Status',
    labels={'Count': 'Number of People', 'Exposure': 'Exposure Status'}
)

# Update layout
fig.update_layout(height=500)

# Store the figure in output_vars to display it
output_vars['fig'] = fig

# Create a bar chart comparing the different measures
measures_df = pd.DataFrame({
    'Measure': ['Risk Ratio', 'Odds Ratio', 'Risk Difference x10'],
    'Value': [risk_ratio, odds_ratio, risk_difference * 10]  # Multiply RD by 10 to make it visible on same scale
})

fig2 = px.bar(
    measures_df,
    x='Measure',
    y='Value',
    title='Comparison of Measures of Association',
    text_auto='.2f',
    color='Measure',
    color_discrete_map={
        'Risk Ratio': 'blue',
        'Odds Ratio': 'green',
        'Risk Difference x10': 'red'
    }
)

# Add a reference line at y=1 (no effect for ratio measures)
fig2.add_shape(
    type="line",
    x0=-0.5, y0=1, x1=2.5, y1=1,
    line=dict(color="black", width=1, dash="dash")
)

fig2.update_layout(
    height=500,
    showlegend=False,
    yaxis_title='Value',
)

# Store the second figure in output_vars
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
            - Change the values in the 2x2 table (`exposed_cases`, `exposed_noncases`, etc.)
            - Create a scenario where the risk ratio and odds ratio are very different
            - Make the exposure protective (RR < 1) by changing the values
            - Try a scenario with a very rare disease (small number of cases)
            
            **Simple Challenges:**
            1. Create a situation where RR = 2.0 (exposure doubles the risk)
            2. Make a "null" scenario where there's no association (RR ≈ 1)
            3. Find combinations where the risk ratio and odds ratio are almost equal
            4. Create a strong protective effect (RR < 0.5)
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
    ### Understanding Measures of Association:
    
    1. **Risk Ratio (RR)**:
       - Divides the risk in the exposed group by the risk in the unexposed group
       - **RR = 1**: No association (equal risk in both groups)
       - **RR > 1**: Exposure increases risk (harmful)
       - **RR < 1**: Exposure decreases risk (protective)
       - Example: RR = 2 means exposed people have twice the risk of unexposed people
    
    2. **Odds Ratio (OR)**:
       - Compares the odds of disease between exposed and unexposed groups
       - Interpreted similarly to the risk ratio
       - Very useful in case-control studies where risks can't be directly calculated
       - Approximates the risk ratio when the disease is rare (<10% in both groups)
    
    3. **Risk Difference (RD)**:
       - Subtracts the risk in the unexposed group from the risk in the exposed group
       - Measures the absolute effect of exposure
       - Used to calculate Number Needed to Treat/Harm (NNT/NNH)
       - Example: RD = 0.1 means exposure increases absolute risk by 10 percentage points
    
    4. **Number Needed to Treat/Harm (NNT/NNH)**:
       - The number of people who need to be exposed to cause or prevent one additional case
       - Calculated as 1 / |Risk Difference|
       - Useful for clinical and public health decision-making
    
    5. **When to Use Each Measure**:
       - **Risk Ratio**: Best for cohort studies and communicating relative effects
       - **Odds Ratio**: Best for case-control studies and rare diseases
       - **Risk Difference**: Best for public health planning and resource allocation
    """)

def comparing_measures_lesson():
    st.header("Comparing Measures of Association")
    
    st.markdown("""
    ### How Different Measures Compare
    
    Different measures of association can lead to different interpretations of the same data.
    This exercise explores how these measures behave under different scenarios and disease frequencies.
    """)
    
    # Initial code example - simplified for beginners
    initial_code = """# Comparing different measures of association
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
disease_prevalence = 0.1       # Overall disease frequency (0.1 = 10%)
true_risk_ratio = 2.0          # True risk ratio we want to simulate
population_size = 1000         # Total population size

# Let's create several scenarios with the same risk ratio but different prevalences
prevalence_levels = [0.01, 0.05, 0.1, 0.2, 0.4]  # From rare (1%) to common (40%)

# Function to calculate a 2x2 table given prevalence, risk ratio, and population size
def calculate_2x2_table(prevalence, risk_ratio, population_size):
    # Assume 50% exposed in the population
    exposed_size = population_size // 2
    unexposed_size = population_size - exposed_size
    
    # Calculate risk in unexposed (baseline risk)
    # We know: 
    # overall_prevalence = (risk_exposed * exposed_size + risk_unexposed * unexposed_size) / population_size
    # risk_exposed = risk_ratio * risk_unexposed
    # Solving for risk_unexposed:
    risk_unexposed = prevalence / ((risk_ratio * exposed_size + unexposed_size) / population_size)
    
    # Ensure risk_unexposed is not too high
    risk_unexposed = min(risk_unexposed, 0.95)
    
    # Calculate risk in exposed
    risk_exposed = risk_ratio * risk_unexposed
    
    # Ensure risk_exposed is not above 1
    risk_exposed = min(risk_exposed, 0.95)
    
    # Calculate the 2x2 table cells
    exposed_cases = round(risk_exposed * exposed_size)
    unexposed_cases = round(risk_unexposed * unexposed_size)
    exposed_noncases = exposed_size - exposed_cases
    unexposed_noncases = unexposed_size - unexposed_cases
    
    # Ensure no zeros in the table (add 1 if needed)
    if exposed_cases == 0:
        exposed_cases = 1
        exposed_noncases -= 1
    if unexposed_cases == 0:
        unexposed_cases = 1
        unexposed_noncases -= 1
    if exposed_noncases == 0:
        exposed_noncases = 1
    if unexposed_noncases == 0:
        unexposed_noncases = 1
    
    return {
        'exposed_cases': exposed_cases,
        'exposed_noncases': exposed_noncases,
        'unexposed_cases': unexposed_cases,
        'unexposed_noncases': unexposed_noncases
    }

# Calculate measures of association for a 2x2 table
def calculate_measures(table):
    exposed_cases = table['exposed_cases']
    exposed_noncases = table['exposed_noncases']
    unexposed_cases = table['unexposed_cases']
    unexposed_noncases = table['unexposed_noncases']
    
    # Calculate risks
    risk_exposed = exposed_cases / (exposed_cases + exposed_noncases)
    risk_unexposed = unexposed_cases / (unexposed_cases + unexposed_noncases)
    
    # Calculate odds
    odds_exposed = exposed_cases / exposed_noncases
    odds_unexposed = unexposed_cases / unexposed_noncases
    
    # Calculate measures
    risk_ratio = risk_exposed / risk_unexposed
    odds_ratio = odds_exposed / odds_unexposed
    risk_difference = risk_exposed - risk_unexposed
    
    # Overall prevalence
    prevalence = (exposed_cases + unexposed_cases) / (exposed_cases + exposed_noncases + unexposed_cases + unexposed_noncases)
    
    return {
        'risk_ratio': risk_ratio,
        'odds_ratio': odds_ratio,
        'risk_difference': risk_difference,
        'prevalence': prevalence,
        'risk_exposed': risk_exposed,
        'risk_unexposed': risk_unexposed
    }

# Calculate measures for each prevalence level
results = []
for prev in prevalence_levels:
    table = calculate_2x2_table(prev, true_risk_ratio, population_size)
    measures = calculate_measures(table)
    
    # Add data to results
    results.append({
        'prevalence': prev,
        'risk_ratio': measures['risk_ratio'],
        'odds_ratio': measures['odds_ratio'],
        'risk_difference': measures['risk_difference'],
        'exposed_cases': table['exposed_cases'],
        'exposed_noncases': table['exposed_noncases'],
        'unexposed_cases': table['unexposed_cases'],
        'unexposed_noncases': table['unexposed_noncases'],
        'risk_exposed': measures['risk_exposed'],
        'risk_unexposed': measures['risk_unexposed']
    })

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Print the results table
print(f"Comparison of Measures with Risk Ratio = {true_risk_ratio}")
print("\\nDisease Prevalence | Risk Ratio | Odds Ratio | Risk Difference")
print("-" * 65)

for _, row in results_df.iterrows():
    print(f"{row['prevalence']*100:16.1f}% | {row['risk_ratio']:9.2f} | {row['odds_ratio']:10.2f} | {row['risk_difference']*100:14.1f}%")

print("\\n2x2 Tables for Each Scenario:")
print("-" * 65)

for i, row in results_df.iterrows():
    print(f"\\nScenario {i+1}: Disease Prevalence = {row['prevalence']*100:.1f}%")
    table = pd.DataFrame({
        'Cases': [row['exposed_cases'], row['unexposed_cases']],
        'Non-cases': [row['exposed_noncases'], row['unexposed_noncases']]
    }, index=['Exposed', 'Unexposed'])
    print(table)
    print(f"Risk in Exposed: {row['risk_exposed']*100:.1f}%, Risk in Unexposed: {row['risk_unexposed']*100:.1f}%")

# Create a graph comparing measures across prevalence levels
fig = go.Figure()

# Add risk ratio line
fig.add_trace(
    go.Scatter(
        x=results_df['prevalence'],
        y=results_df['risk_ratio'],
        mode='lines+markers',
        name='Risk Ratio',
        line=dict(color='blue', width=3)
    )
)

# Add odds ratio line
fig.add_trace(
    go.Scatter(
        x=results_df['prevalence'],
        y=results_df['odds_ratio'],
        mode='lines+markers',
        name='Odds Ratio',
        line=dict(color='green', width=3)
    )
)

# Add risk difference line (multiplied for scale)
fig.add_trace(
    go.Scatter(
        x=results_df['prevalence'],
        y=results_df['risk_difference'] * 10,  # Scale up for visibility
        mode='lines+markers',
        name='Risk Difference x10',
        line=dict(color='red', width=3)
    )
)

# Add a reference line at y=true_risk_ratio
fig.add_shape(
    type="line",
    x0=0, y0=true_risk_ratio, x1=max(prevalence_levels), y1=true_risk_ratio,
    line=dict(color="black", width=1, dash="dash")
)

# Update layout
fig.update_layout(
    title=f'Measures of Association with Risk Ratio = {true_risk_ratio}',
    xaxis_title='Disease Prevalence',
    yaxis_title='Measure Value',
    height=500,
    xaxis=dict(tickformat='.0%'),
    legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
)

# Store the figure in output_vars
output_vars['fig'] = fig

# Create a second figure showing risk difference
fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=[f"{p*100:.0f}%" for p in results_df['prevalence']],
        y=results_df['risk_difference'] * 100,  # Convert to percentage
        text=[f"{rd*100:.1f}%" for rd in results_df['risk_difference']],
        textposition='auto',
        marker_color='red',
        name='Risk Difference (%)'
    )
)

fig2.update_layout(
    title='Risk Difference by Disease Prevalence',
    xaxis_title='Disease Prevalence',
    yaxis_title='Risk Difference (%)',
    height=400
)

# Store the second figure
output_vars['fig2'] = fig2

# Create a third figure showing risks in exposed and unexposed
fig3 = go.Figure()

# Add exposed risk line
fig3.add_trace(
    go.Scatter(
        x=results_df['prevalence'],
        y=results_df['risk_exposed'],
        mode='lines+markers',
        name='Risk in Exposed',
        line=dict(color='red', width=3)
    )
)

# Add unexposed risk line
fig3.add_trace(
    go.Scatter(
        x=results_df['prevalence'],
        y=results_df['risk_unexposed'],
        mode='lines+markers',
        name='Risk in Unexposed',
        line=dict(color='blue', width=3)
    )
)

# Update layout
fig3.update_layout(
    title='Risk by Exposure Status Across Disease Prevalence',
    xaxis_title='Disease Prevalence',
    yaxis_title='Risk',
    height=400,
    xaxis=dict(tickformat='.0%'),
    yaxis=dict(tickformat='.0%')
)

# Store the third figure
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
            - Change `true_risk_ratio` to different values (try 1.5, 3.0, or 0.5)
            - Adjust `prevalence_levels` to explore different disease frequencies
            - Increase `population_size` to see if it affects the stability of the measures
            
            **Simple Challenges:**
            1. Find when the risk ratio and odds ratio are very different
            2. Try a protective exposure by setting `true_risk_ratio` to 0.5
            3. What happens when the disease becomes very common (prevalence > 30%)?
            4. Look at how risk difference changes with disease prevalence
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
    
    # Include a discussion section - simplified for beginners
    st.subheader("Key Concepts")
    st.markdown("""
    ### How Measures Differ:
    
    1. **Relationship Between Risk Ratio (RR) and Odds Ratio (OR)**:
       - When a disease is rare (prevalence < 10%), the OR closely approximates the RR
       - As disease prevalence increases, OR and RR can differ substantially
       - For common diseases, OR tends to be further from 1 than RR (more extreme)
       - Example: For a common disease, an RR of 2 might correspond to an OR of 4 or more
    
    2. **Behavior of Risk Difference (RD)**:
       - Risk difference tends to increase with disease prevalence, even with the same RR
       - The maximum possible RD depends on the baseline risk
       - Example: If baseline risk is 1%, max RD is 99%; if baseline risk is 40%, max RD is 60%
    
    3. **Strengths of Each Measure**:
       - **Risk Ratio**: Easy to interpret, consistent across populations with different baseline risks
       - **Odds Ratio**: Can be calculated in any study design, useful for statistical modeling
       - **Risk Difference**: Directly measures public health impact, useful for resource allocation
    
    4. **Limitations of Each Measure**:
       - **Risk Ratio**: Doesn't consider baseline risk, difficult to calculate in case-control studies
       - **Odds Ratio**: Can be misinterpreted as a risk ratio, becomes misleading for common diseases
       - **Risk Difference**: Varies with baseline risk, may not be transferable to other populations
    
    5. **Choosing the Right Measure**:
       - Use **multiple measures** to get a complete picture
       - Consider your **study design** (cohort vs. case-control)
       - Think about your **audience** (clinicians, public health officials, patients)
       - Remember the **disease frequency** in your population
    """)

if __name__ == "__main__":
    app()