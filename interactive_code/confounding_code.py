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
    st.title("Confounding: Interactive Code Laboratory")
    
    st.markdown("""
    ## Learn by Coding: Confounding in Epidemiology
    
    This interactive laboratory helps you understand how confounding works through simple
    Python code examples. Confounding is when a third variable (a confounder) affects both
    the exposure and outcome, creating a misleading relationship between them.
    
    Choose an example below to get started:
    """)
    
    # Example selection
    example = st.selectbox(
        "Select an example:",
        ["Age, Smoking, and Cancer Risk", 
         "Education, Exercise, and Health Outcomes",
         "Caffeine, Study Time, and Exam Scores"]
    )
    
    # Display the selected example
    if example == "Age, Smoking, and Cancer Risk":
        age_smoking_cancer_example()
    elif example == "Education, Exercise, and Health Outcomes":
        education_exercise_health_example()
    elif example == "Caffeine, Study Time, and Exam Scores":
        caffeine_study_scores_example()

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

def age_smoking_cancer_example():
    st.header("Age, Smoking, and Cancer Risk")
    
    st.markdown("""
    ### Understanding Confounding in Smoking-Cancer Studies
    
    In epidemiological studies of smoking and cancer, **age** is an important confounder because:
    
    1. **Age affects smoking history**: Older individuals have had more time to accumulate pack-years
    2. **Age affects cancer risk**: Cancer risk increases with age for biological reasons
    
    If we don't account for age, we might **overestimate** the effect of smoking on cancer risk.
    
    Let's simulate this scenario and see how to detect and adjust for confounding:
    """)
    
    # Initial code example
    initial_code = """# Simulating confounding in the relationship between smoking and cancer risk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# MODIFY THESE PARAMETERS TO SEE DIFFERENT EFFECTS
confounder_strength = 0.6  # How strongly age affects both smoking and cancer risk
true_effect = 0.3         # The true causal effect of smoking on cancer risk
n_subjects = 200          # Number of subjects in the study

# Generate data with confounding
def generate_confounded_data():
    # Step 1: Generate age (confounder)
    # Ages between 30 and 80 years
    age = np.random.normal(55, 15, n_subjects)
    age = np.clip(age, 30, 80)
    
    # Step 2: Generate smoking history (affected by age)
    # Smoking history in pack-years
    smoking_base = np.random.normal(0, 10, n_subjects)
    smoking = age * confounder_strength/10 + smoking_base
    smoking = np.clip(smoking, 0, None)  # No negative smoking
    
    # Step 3: Generate cancer risk (affected by both age and smoking)
    # Cancer risk score (higher = higher risk)
    cancer_base = np.random.normal(0, 2, n_subjects)
    cancer_risk = (
        smoking * true_effect + 
        age * confounder_strength/10 + 
        cancer_base
    )
    
    # Create dataframe
    df = pd.DataFrame({
        'Age': age,
        'Smoking_PackYears': smoking,
        'Cancer_Risk': cancer_risk
    })
    
    return df

# Generate the data
data = generate_confounded_data()

# Print summary statistics
print("Data Summary (first 5 rows):")
print(data.head())

print("\\nSummary Statistics:")
print(data.describe().round(2))

# Calculate correlations
corr_smoking_cancer = data['Smoking_PackYears'].corr(data['Cancer_Risk'])
corr_age_smoking = data['Age'].corr(data['Smoking_PackYears'])
corr_age_cancer = data['Age'].corr(data['Cancer_Risk'])

print("\\nCorrelations:")
print(f"Smoking-Cancer: {corr_smoking_cancer:.3f}")
print(f"Age-Smoking: {corr_age_smoking:.3f}")
print(f"Age-Cancer: {corr_age_cancer:.3f}")

# Create scatter plot: Smoking vs Cancer (Crude/Unadjusted)
fig1 = px.scatter(
    data, 
    x='Smoking_PackYears', 
    y='Cancer_Risk',
    title='Smoking vs Cancer Risk (Unadjusted)',
    trendline='ols'  # Add ordinary least squares regression line
)

# Use age as color to visualize the confounding
fig1.update_traces(
    marker=dict(
        size=8,
        color=data['Age'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Age (years)")
    )
)

# Update layout
fig1.update_layout(
    xaxis_title='Smoking History (Pack-Years)',
    yaxis_title='Cancer Risk Score'
)

# Store the figure
output_vars['fig1'] = fig1

# Fit regression models
# 1. Crude model (unadjusted for age)
from statsmodels.formula.api import ols
crude_model = ols('Cancer_Risk ~ Smoking_PackYears', data=data).fit()
crude_effect = crude_model.params['Smoking_PackYears']

# 2. Adjusted model (controlling for age)
adjusted_model = ols('Cancer_Risk ~ Smoking_PackYears + Age', data=data).fit()
adjusted_effect = adjusted_model.params['Smoking_PackYears']

print("\\nRegression Results:")
print(f"Crude effect (not controlling for age): {crude_effect:.3f}")
print(f"Adjusted effect (controlling for age): {adjusted_effect:.3f}")
print(f"True effect (simulation parameter): {true_effect:.3f}")
print(f"Confounding bias: {crude_effect - adjusted_effect:.3f}")

# Stratified analysis
# Create age groups (strata)
data['Age_Group'] = pd.cut(
    data['Age'], 
    bins=[30, 45, 60, 80], 
    labels=['30-45', '46-60', '61-80']
)

# Create stratified scatter plot
fig2 = px.scatter(
    data, 
    x='Smoking_PackYears', 
    y='Cancer_Risk', 
    color='Age_Group',
    facet_col='Age_Group',
    trendline='ols',
    title='Stratified Analysis: Smoking vs Cancer Risk by Age Group'
)

# Update layout
fig2.update_layout(
    xaxis_title='Smoking History (Pack-Years)',
    yaxis_title='Cancer Risk Score'
)

# Store the stratified figure
output_vars['fig2'] = fig2

# Create a visualization of the confounding relationships
# Simple directed acyclic graph (DAG)
fig3 = go.Figure()

# Define node positions
nodes = {
    'Age': {'x': 0, 'y': 1},
    'Smoking': {'x': -1, 'y': 0},
    'Cancer': {'x': 1, 'y': 0}
}

# Add nodes
for node, pos in nodes.items():
    fig3.add_trace(go.Scatter(
        x=[pos['x']], 
        y=[pos['y']],
        mode='markers+text',
        marker=dict(size=40, color='lightblue', line=dict(color='black', width=1)),
        text=node,
        textposition="middle center",
        hoverinfo='text',
        showlegend=False
    ))

# Add edges with labels
edges = [
    {'from': 'Age', 'to': 'Smoking', 'label': f'{corr_age_smoking:.2f}'},
    {'from': 'Age', 'to': 'Cancer', 'label': f'{corr_age_cancer:.2f}'},
    {'from': 'Smoking', 'to': 'Cancer', 'label': f'{corr_smoking_cancer:.2f} (crude)\\n{adjusted_effect:.2f} (adj)'}
]

for edge in edges:
    # Get positions
    x0, y0 = nodes[edge['from']]['x'], nodes[edge['from']]['y']
    x1, y1 = nodes[edge['to']]['x'], nodes[edge['to']]['y']
    
    # Add line
    fig3.add_trace(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode='lines',
        line=dict(width=2, color='grey'),
        showlegend=False
    ))
    
    # Add label at midpoint
    fig3.add_annotation(
        x=(x0 + x1) / 2,
        y=(y0 + y1) / 2,
        text=edge['label'],
        showarrow=False,
        bgcolor="white"
    )

# Update layout
fig3.update_layout(
    title='Confounding Relationship Diagram',
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-1.5, 1.5]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.5, 1.5]
    )
)

# Store the DAG figure
output_vars['fig3'] = fig3
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
        with st.expander("Try Yourself"):
            st.markdown("""
            **Try modifying:**
            - Change `confounder_strength` to see how the strength of confounding affects the results
            - Set `true_effect` to zero to see how confounding can create an apparent relationship when none exists
            - Try creating negative confounding by making age negatively related to either smoking or cancer risk
            
            **Challenge yourself:**
            1. Add another confounder, like socioeconomic status, that affects both smoking and cancer risk
            2. Implement propensity score matching as another way to adjust for confounding
            3. Create a scenario where the crude effect underestimates the true effect
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
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Insights")
    st.markdown("""
    ### What This Analysis Shows:
    
    1. **Confounding in Action**: 
       - The unadjusted (crude) relationship between smoking and cancer risk is stronger than the true causal effect
       - This is because age increases both smoking duration and cancer risk independently
    
    2. **Detecting Confounding**:
       - Look for variables that correlate with both the exposure and outcome
       - Check if the crude effect changes substantially after adjustment
       - Examine whether the relationship varies across strata of the confounder
    
    3. **Adjusting for Confounding**:
       - **Statistical adjustment**: Adding the confounder to regression models
       - **Stratification**: Analyzing the relationship within similar age groups
       - **Other methods**: Matching, restriction, propensity scores (not shown here)
    
    ### Clinical Implications:
    
    - **Risk Assessment**: Without adjusting for age, we might overestimate a patient's cancer risk based on smoking history
    - **Research Design**: Always consider potential confounders when designing and analyzing studies
    - **Causal Inference**: Don't assume associations represent causal relationships without addressing confounding
    
    Remember that in real epidemiological studies, there may be multiple confounders that need to be addressed simultaneously!
    """)

def education_exercise_health_example():
    st.header("Education, Exercise, and Health Outcomes")
    
    st.markdown("""
    ### Understanding Confounding in Exercise-Health Studies
    
    When studying the relationship between exercise and health outcomes, **education level** can act as a 
    confounder because:
    
    1. **Education affects exercise habits**: More educated people tend to exercise more
    2. **Education affects health outcomes**: Higher education is associated with better health for many reasons
    
    Without accounting for education, we might **overattribute** health benefits to exercise alone.
    
    Let's simulate this scenario and explore confounding:
    """)
    
    # Initial code example
    initial_code = """# Simulating confounding in the relationship between exercise and health
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set random seed for reproducibility
np.random.seed(123)

# MODIFY THESE PARAMETERS TO SEE DIFFERENT EFFECTS
confounder_strength = 0.5  # How strongly education affects exercise and health
true_effect = 0.4         # The true causal effect of exercise on health
n_subjects = 250          # Number of subjects in the study

# Generate data with confounding
def generate_confounded_data():
    # Step 1: Generate education level (confounder)
    # Education in years (8-20 years)
    education = np.random.normal(14, 3, n_subjects)
    education = np.clip(education, 8, 20)
    
    # Step 2: Generate exercise frequency (affected by education)
    # Exercise in hours per week
    exercise_base = np.random.normal(0, 2, n_subjects)
    exercise = education * confounder_strength/5 + exercise_base
    exercise = np.clip(exercise, 0, 14)  # Limit to realistic values
    
    # Step 3: Generate health outcome (affected by both education and exercise)
    # Health score (higher = better health)
    health_base = np.random.normal(0, 1, n_subjects)
    health = (
        exercise * true_effect + 
        education * confounder_strength/3 + 
        health_base
    )
    health = np.clip(health, 1, 10)  # Health score from 1-10
    
    # Create dataframe
    df = pd.DataFrame({
        'Education_Years': education,
        'Exercise_Hours': exercise,
        'Health_Score': health
    })
    
    return df

# Generate the data
data = generate_confounded_data()

# Print summary statistics
print("Data Summary (first 5 rows):")
print(data.head())

print("\\nSummary Statistics:")
print(data.describe().round(2))

# Calculate correlations
corr_exercise_health = data['Exercise_Hours'].corr(data['Health_Score'])
corr_edu_exercise = data['Education_Years'].corr(data['Exercise_Hours'])
corr_edu_health = data['Education_Years'].corr(data['Health_Score'])

print("\\nCorrelations:")
print(f"Exercise-Health: {corr_exercise_health:.3f}")
print(f"Education-Exercise: {corr_edu_exercise:.3f}")
print(f"Education-Health: {corr_edu_health:.3f}")

# Create scatter plot: Exercise vs Health (Crude/Unadjusted)
fig1 = px.scatter(
    data, 
    x='Exercise_Hours', 
    y='Health_Score',
    title='Exercise vs Health (Unadjusted)',
    trendline='ols'  # Add ordinary least squares regression line
)

# Use education as color to visualize the confounding
fig1.update_traces(
    marker=dict(
        size=8,
        color=data['Education_Years'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Education (years)")
    )
)

# Update layout
fig1.update_layout(
    xaxis_title='Exercise (Hours/Week)',
    yaxis_title='Health Score (1-10)'
)

# Store the figure
output_vars['fig1'] = fig1

# Fit regression models
# 1. Crude model (unadjusted for education)
from statsmodels.formula.api import ols
crude_model = ols('Health_Score ~ Exercise_Hours', data=data).fit()
crude_effect = crude_model.params['Exercise_Hours']

# 2. Adjusted model (controlling for education)
adjusted_model = ols('Health_Score ~ Exercise_Hours + Education_Years', data=data).fit()
adjusted_effect = adjusted_model.params['Exercise_Hours']

print("\\nRegression Results:")
print(f"Crude effect (not controlling for education): {crude_effect:.3f}")
print(f"Adjusted effect (controlling for education): {adjusted_effect:.3f}")
print(f"True effect (simulation parameter): {true_effect:.3f}")
print(f"Confounding bias: {crude_effect - adjusted_effect:.3f}")

# Stratified analysis
# Create education groups (strata)
data['Education_Group'] = pd.cut(
    data['Education_Years'], 
    bins=[8, 12, 16, 20], 
    labels=['8-12 yrs', '13-16 yrs', '17-20 yrs']
)

# Create stratified scatter plot
fig2 = px.scatter(
    data, 
    x='Exercise_Hours', 
    y='Health_Score', 
    color='Education_Group',
    facet_col='Education_Group',
    trendline='ols',
    title='Stratified Analysis: Exercise vs Health by Education Level'
)

# Update layout
fig2.update_layout(
    xaxis_title='Exercise (Hours/Week)',
    yaxis_title='Health Score (1-10)'
)

# Store the stratified figure
output_vars['fig2'] = fig2

# Calculate average health score by exercise category and education group
# Create exercise categories
data['Exercise_Category'] = pd.cut(
    data['Exercise_Hours'],
    bins=[0, 2, 5, 14],
    labels=['Low', 'Medium', 'High']
)

# Calculate means
grouped_means = data.groupby(['Education_Group', 'Exercise_Category'])['Health_Score'].mean().reset_index()

# Create grouped bar chart
fig3 = px.bar(
    grouped_means,
    x='Exercise_Category',
    y='Health_Score',
    color='Education_Group',
    barmode='group',
    title='Average Health Score by Exercise Level and Education'
)

# Update layout
fig3.update_layout(
    xaxis_title='Exercise Level',
    yaxis_title='Average Health Score',
    xaxis={'categoryorder':'array', 'categoryarray':['Low', 'Medium', 'High']}
)

# Store the bar chart
output_vars['fig3'] = fig3
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
        with st.expander("Try Yourself"):
            st.markdown("""
            **Try modifying:**
            - Adjust `confounder_strength` to see how education's influence changes the results
            - Try increasing `true_effect` to make exercise more important to health
            - Change the number of subjects to see how sample size affects your conclusions
            
            **Challenge yourself:**
            1. Add a second health outcome that is affected differently by exercise and education
            2. Create interaction effects where exercise benefits differ by education level
            3. Implement a matching procedure where each person is matched with someone of similar education
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
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Insights")
    st.markdown("""
    ### What This Analysis Shows:
    
    1. **Education as a Confounder**: 
       - People with more education tend to exercise more AND have better health for many reasons
       - Without adjustment, we attribute too much of the health benefit to exercise itself
    
    2. **Multiple Methods to Address Confounding**:
       - **Regression adjustment**: Isolates the effect of exercise by controlling for education
       - **Stratification**: Examines the exercise-health relationship within education groups
       - **Examining averages**: Comparing health scores across combined categories
    
    3. **Public Health Implications**:
       - Exercise does have real health benefits (the true effect)
       - But addressing educational disparities might have additional health benefits
       - Exercise promotion should be culturally appropriate for all education levels
    
    ### Research Design Considerations:
    
    - **Measuring confounders**: Always collect data on potential confounders (like education)
    - **Causal assumptions**: Be clear about what variables you believe affect each other
    - **Effect estimation**: Report both crude and adjusted effects for transparency
    - **Target population**: Consider whether results generalize to populations with different education distributions
    
    These principles apply broadly across epidemiological and health services research!
    """)

def caffeine_study_scores_example():
    st.header("Caffeine, Study Time, and Exam Scores")
    
    st.markdown("""
    ### Understanding Confounding in a Simple Everyday Example
    
    This example uses a relatable scenario for students: the relationship between caffeine consumption,
    study time, and exam scores. 
    
    In this case, **study time** is a confounder because:
    
    1. **Study time affects caffeine consumption**: Students who study more tend to drink more coffee/tea
    2. **Study time affects exam scores**: More studying generally leads to better scores
    
    If we don't account for study time, we might incorrectly conclude that **caffeine improves exam scores**
    when the real driver is study time.
    
    Let's simulate this scenario:
    """)
    
    # Initial code example
    initial_code = """# Simulating confounding in the relationship between caffeine and exam scores
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set random seed for reproducibility
np.random.seed(456)

# MODIFY THESE PARAMETERS TO SEE DIFFERENT EFFECTS
confounder_strength = 0.7  # How strongly study time affects caffeine and scores
true_effect = 0.1         # The true causal effect of caffeine on exam scores (small or none)
n_students = 150          # Number of students in the study

# Generate data with confounding
def generate_confounded_data():
    # Step 1: Generate study time (confounder)
    # Study time in hours per week
    study_time = np.random.gamma(shape=2, scale=5, size=n_students)  # Right-skewed
    study_time = np.clip(study_time, 0, 40)
    
    # Step 2: Generate caffeine consumption (affected by study time)
    # Caffeine in mg per day
    caffeine_base = np.random.normal(0, 50, n_students)
    caffeine = study_time * confounder_strength * 10 + caffeine_base + 100
    caffeine = np.clip(caffeine, 0, 500)  # Realistic caffeine intake
    
    # Step 3: Generate exam scores (affected by study time and slightly by caffeine)
    # Scores from 0-100
    score_base = np.random.normal(0, 10, n_students)
    exam_score = (
        study_time * confounder_strength * 2 + 
        caffeine * true_effect / 100 +  # Small or no true effect
        score_base + 60  # Base score around 60-70
    )
    exam_score = np.clip(exam_score, 0, 100)
    
    # Create dataframe
    df = pd.DataFrame({
        'Study_Hours': study_time,
        'Caffeine_mg': caffeine,
        'Exam_Score': exam_score
    })
    
    return df

# Generate the data
data = generate_confounded_data()

# Print summary statistics
print("Data Summary (first 5 rows):")
print(data.head())

print("\\nSummary Statistics:")
print(data.describe().round(2))

# Calculate correlations
corr_caffeine_score = data['Caffeine_mg'].corr(data['Exam_Score'])
corr_study_caffeine = data['Study_Hours'].corr(data['Caffeine_mg'])
corr_study_score = data['Study_Hours'].corr(data['Exam_Score'])

print("\\nCorrelations:")
print(f"Caffeine-Exam Score: {corr_caffeine_score:.3f}")
print(f"Study Time-Caffeine: {corr_study_caffeine:.3f}")
print(f"Study Time-Exam Score: {corr_study_score:.3f}")

# Create scatter plot: Caffeine vs Exam Score (Crude/Unadjusted)
fig1 = px.scatter(
    data, 
    x='Caffeine_mg', 
    y='Exam_Score',
    title='Caffeine vs Exam Score (Unadjusted)',
    trendline='ols'  # Add ordinary least squares regression line
)

# Use study time as color to visualize the confounding
fig1.update_traces(
    marker=dict(
        size=8,
        color=data['Study_Hours'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Study Hours/Week")
    )
)

# Update layout
fig1.update_layout(
    xaxis_title='Caffeine Consumption (mg/day)',
    yaxis_title='Exam Score (0-100)'
)

# Store the figure
output_vars['fig1'] = fig1

# Fit regression models
# 1. Crude model (unadjusted for study time)
from statsmodels.formula.api import ols
crude_model = ols('Exam_Score ~ Caffeine_mg', data=data).fit()
crude_effect = crude_model.params['Caffeine_mg']

# 2. Adjusted model (controlling for study time)
adjusted_model = ols('Exam_Score ~ Caffeine_mg + Study_Hours', data=data).fit()
adjusted_effect = adjusted_model.params['Caffeine_mg']

print("\\nRegression Results:")
print(f"Crude effect (not controlling for study time): {crude_effect:.5f}")
print(f"Adjusted effect (controlling for study time): {adjusted_effect:.5f}")
print(f"True effect (simulation parameter): {true_effect/100:.5f}")
print(f"Confounding bias: {crude_effect - adjusted_effect:.5f}")

# Stratified analysis
# Create study time groups (strata)
data['Study_Group'] = pd.cut(
    data['Study_Hours'], 
    bins=[0, 10, 20, 40], 
    labels=['Low (0-10h)', 'Medium (11-20h)', 'High (21-40h)']
)

# Create stratified scatter plot
fig2 = px.scatter(
    data, 
    x='Caffeine_mg', 
    y='Exam_Score', 
    color='Study_Group',
    facet_col='Study_Group',
    trendline='ols',
    title='Stratified Analysis: Caffeine vs Exam Score by Study Time'
)

# Update layout
fig2.update_layout(
    xaxis_title='Caffeine Consumption (mg/day)',
    yaxis_title='Exam Score (0-100)'
)

# Store the stratified figure
output_vars['fig2'] = fig2

# Create a 3D visualization to see all three variables
fig3 = px.scatter_3d(
    data,
    x='Caffeine_mg',
    y='Study_Hours',
    z='Exam_Score',
    color='Study_Hours',
    title='3D Visualization: Caffeine, Study Time, and Exam Scores'
)

# Update layout
fig3.update_layout(
    scene=dict(
        xaxis_title='Caffeine (mg/day)',
        yaxis_title='Study Time (hours/week)',
        zaxis_title='Exam Score'
    )
)

# Store the 3D figure
output_vars['fig3'] = fig3
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
        with st.expander("Try Yourself"):
            st.markdown("""
            **Try modifying:**
            - Set `true_effect` to 0 to make caffeine have no real effect on exam scores
            - Try setting `true_effect` to a negative value (caffeine might actually harm scores at high levels)
            - Adjust `confounder_strength` to see how the study time relationship changes the results
            
            **Challenge yourself:**
            1. Add another variable like "sleep quality" that is negatively affected by caffeine and positively affects scores
            2. Add random measurement error to the caffeine variable to simulate imperfect recall
            3. Create a "Simpson's Paradox" scenario where the overall relationship is positive but within each stratum it's negative
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
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section
    st.subheader("Key Insights")
    st.markdown("""
    ### What This Analysis Shows:
    
    1. **Study Time as a Confounder**: 
       - Students who study more tend to consume more caffeine AND get better exam scores
       - This creates an apparent relationship between caffeine and scores
       - After adjustment, we can see that caffeine has little or no effect on its own
    
    2. **Detecting Spurious Relationships**:
       - The crude analysis suggests caffeine improves scores
       - The stratified analysis shows little effect within each study time group
       - The 3D plot helps visualize how study time drives both variables
    
    3. **Statistical vs. Causal Interpretation**:
       - Statistical association: Caffeine and scores are positively correlated
       - Causal interpretation: Caffeine has minimal effect on scores after accounting for study time
       - This distinction is crucial for making correct recommendations
    
    ### Practical Applications:
    
    - **Student Advice**: Focus on study time, not coffee consumption, to improve grades
    - **Research Design**: Always consider potential confounders when designing studies
    - **Critical Thinking**: Question whether observed associations might be explained by third factors
    
    This simple example illustrates how confounding can lead to incorrect conclusions in many fields,
    from medicine to marketing to social science.
    """)

if __name__ == "__main__":
    app()