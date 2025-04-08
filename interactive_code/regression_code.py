import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split
import io
from contextlib import redirect_stdout
import sys
import traceback

def app():
    st.title("Interactive Coding Laboratory: Regression Analysis")
    
    st.markdown("""
    ## Learn by Coding: Regression Analysis for Pharmacy and Epidemiology
    
    This interactive coding laboratory allows you to modify and execute Python code directly in your browser.
    Experiment with different regression methods by modifying the example code and seeing the results.
    
    Choose a topic to explore:
    """)
    
    # Topic selection
    topic = st.selectbox(
        "Select a regression topic:",
        ["Basic Linear Regression", 
         "Multiple Linear Regression",
         "Basic Logistic Regression"]
    )
    
    # Display the selected topic
    if topic == "Basic Linear Regression":
        basic_linear_regression_lesson()
    elif topic == "Multiple Linear Regression":
        multiple_linear_regression_lesson()
    elif topic == "Basic Logistic Regression":
        basic_logistic_regression_lesson()


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
                'LinearRegression': LinearRegression,
                'LogisticRegression': LogisticRegression,
                'r2_score': r2_score,
                'mean_squared_error': mean_squared_error,
                'roc_curve': roc_curve, 
                'auc': auc,
                'train_test_split': train_test_split,
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

def basic_linear_regression_lesson():
    st.header("Basic Linear Regression")
    
    st.markdown("""
    ### Understanding Linear Regression
    
    Linear regression models the relationship between a dependent variable (y) and one or more 
    independent variables (x) using a linear equation.
    
    In pharmacy and epidemiology, linear regression can help us understand:
    
    - How medication dosage relates to blood drug concentration
    - The relationship between drug exposure and therapeutic effect
    - How patient factors influence treatment response
    
    Let's explore basic linear regression:
    """)
    
    # Initial code example
    initial_code = """# Basic linear regression
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 50       # Number of data points
effect_size = 0.7    # Strength of relationship (0 to 1)
noise_level = 1.0    # Amount of random variation
    
# Generate simulated drug dosage and concentration data
np.random.seed(42)  # For reproducible results

# Simulate drug dosage (mg)
dosage = np.random.uniform(50, 200, n_samples)

# Simulate blood concentration with some noise (μg/L)
# We expect a linear relationship between dose and concentration
concentration = (10 + effect_size * dosage + 
                np.random.normal(0, noise_level * 10, n_samples))

# Create a DataFrame with the data
drug_data = pd.DataFrame({
    'Dosage_mg': dosage,
    'Concentration_ug_L': concentration
})

# Print the first few rows of data
print("Drug dosage and concentration data:")
print(drug_data.head())

# Calculate basic statistics
print("\\nBasic statistics:")
print(drug_data.describe())

# Calculate Pearson correlation
correlation = np.corrcoef(dosage, concentration)[0, 1]
print(f"\\nCorrelation between dosage and concentration: {correlation:.3f}")

# Perform linear regression
X = dosage.reshape(-1, 1)  # Reshape for scikit-learn
y = concentration

model = LinearRegression()
model.fit(X, y)

# Get model parameters
slope = model.coef_[0]
intercept = model.intercept_

print("\\nLinear Regression Results:")
print(f"Equation: Concentration = {intercept:.2f} + {slope:.2f} × Dosage")

# Make predictions
y_pred = model.predict(X)

# Calculate goodness of fit
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R² (Coefficient of determination): {r2:.3f}")
print(f"RMSE (Root Mean Square Error): {rmse:.3f} μg/L")

# Create a scatter plot with regression line
fig = px.scatter(
    drug_data, x='Dosage_mg', y='Concentration_ug_L',
    title=f"Relationship Between Drug Dosage and Blood Concentration (R² = {r2:.3f})",
    labels={
        'Dosage_mg': 'Drug Dosage (mg)',
        'Concentration_ug_L': 'Blood Concentration (μg/L)'
    },
    trendline='ols',  # Add OLS regression line
    trendline_color_override='red'
)

# Update layout
fig.update_layout(
    title={
        'font_size': 20,
        'xanchor': 'center',
        'x': 0.5
    }
)

# Save figure for display
output_vars['fig'] = fig

# PHARMACOKINETIC INTERPRETATION

# Calculate concentration per mg dose
drug_data['Conc_per_mg'] = drug_data['Concentration_ug_L'] / drug_data['Dosage_mg']

# Mean concentration per mg dose
mean_conc_per_mg = drug_data['Conc_per_mg'].mean()

print("\\nPharmacokinetic Interpretation:")
print(f"Mean concentration per mg dose: {mean_conc_per_mg:.3f} μg/L per mg")
print(f"Estimated concentration for 100 mg dose: {intercept + slope * 100:.2f} μg/L")

# Calculate therapeutic range (simplified example)
therapeutic_min = intercept + slope * 75  # Minimum effective dose is 75 mg
therapeutic_max = intercept + slope * 150  # Maximum safe dose is 150 mg

print(f"Estimated therapeutic range: {therapeutic_min:.2f} - {therapeutic_max:.2f} μg/L")

# Interpretation guidance
print("\\nClinical Interpretation:")
if slope > 0.5:
    print("Strong dosage-concentration relationship: Dose adjustments will significantly affect blood levels")
elif slope > 0.2:
    print("Moderate dosage-concentration relationship: Dose adjustments will have a measurable effect on blood levels")
else:
    print("Weak dosage-concentration relationship: Dose adjustments may have limited impact on blood levels")

if r2 > 0.8:
    print("Model has high precision for estimating concentration from dosage")
elif r2 > 0.5:
    print("Model has moderate precision for estimating concentration from dosage")
else:
    print("Model has low precision - consider individual patient variability")

# Create a second figure showing the predicted concentration for a range of doses
dose_range = np.linspace(25, 225, 100)
pred_conc = intercept + slope * dose_range

fig2 = go.Figure()

# Add the therapeutic range as a colored background
fig2.add_shape(
    type="rect",
    x0=75, x1=150,
    y0=therapeutic_min, y1=therapeutic_max,
    fillcolor="lightgreen",
    opacity=0.3,
    line_width=0,
    layer="below"
)

# Add the predicted concentration line
fig2.add_trace(
    go.Scatter(
        x=dose_range,
        y=pred_conc,
        mode="lines",
        name="Predicted Concentration",
        line=dict(color="blue", width=2)
    )
)

# Add original data points
fig2.add_trace(
    go.Scatter(
        x=dosage,
        y=concentration,
        mode="markers",
        name="Observed Data",
        marker=dict(size=8, color="darkblue", opacity=0.6)
    )
)

# Add annotations for therapeutic range
fig2.add_annotation(
    x=112.5, y=therapeutic_max + 10,
    text="Therapeutic Range",
    showarrow=False,
    font=dict(size=14, color="green")
)

# Update layout
fig2.update_layout(
    title="Dosage-Concentration Relationship with Therapeutic Range",
    xaxis_title="Drug Dosage (mg)",
    yaxis_title="Blood Concentration (μg/L)",
    height=500
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
            - Change `effect_size` to see how it affects the relationship strength
            - Increase or decrease `noise_level` to see how it affects model precision
            - Adjust `n_samples` to see how sample size affects result stability
            
            **Challenges:**
            1. Modify the code to simulate a non-linear relationship between dosage and concentration
            2. Add a visualization showing prediction intervals (uncertainty in predictions)
            3. Create a scenario where higher doses have greater variability in concentration
            4. Compare the linear model fit between different subgroups (e.g., high vs. low doses)
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
    ### Linear Regression in Pharmacy:
    
    1. **Model Equation**: Concentration = Intercept + Slope × Dosage
        - Intercept: Expected concentration at zero dose (often not meaningful)
        - Slope: Increase in concentration per unit increase in dose

    2. **Clinical Applications**:
        - Dose-concentration relationships for therapeutic drug monitoring
        - Predicting drug levels for personalized dosing
        - Analyzing bioequivalence of different formulations
        - Determining dose adjustments for renal/hepatic impairment

    3. **Important Metrics**:
        - R²: Measures how well dosage explains variation in concentration
        - RMSE: Typical error in concentration predictions (in μg/L)
        - Slope: Represents the pharmacokinetic parameter relating dose to concentration
    
    4. **Assumptions**:
        - Linear relationship between dose and concentration (may not hold for all drugs)
        - Constant variance across dosage range
        - Independence of observations
        - Normally distributed errors

    5. **Pharmacokinetic Relevance**:
        - Slope represents the increase in concentration per mg of dose
        - Can help establish therapeutic ranges
        - Supports dose adjustments for individual patients
    """)

def multiple_linear_regression_lesson():
    st.header("Multiple Linear Regression")
    
    st.markdown("""
    ### Understanding Multiple Linear Regression
    
    Multiple linear regression extends simple linear regression by including multiple predictor variables:
    
    y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + ε
    
    In pharmacy and epidemiology, this helps us understand how multiple factors simultaneously 
    influence outcomes such as:
    
    - How patient characteristics affect drug metabolism
    - Multiple determinants of treatment response
    - Factors influencing medication adherence
    
    Let's explore multiple linear regression:
    """)
    
    # Initial code example
    initial_code = """# Multiple linear regression
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 100        # Number of samples
age_effect = -0.2      # Effect of age on clearance (negative means clearance decreases with age)
weight_effect = 0.3    # Effect of weight on clearance
kidney_effect = 0.5    # Effect of kidney function on clearance
noise_level = 1.0      # Amount of random variation

# Generate simulated patient data
np.random.seed(42)  # For reproducible results

# Patient characteristics
age = np.random.normal(65, 15, n_samples)      # Age in years
weight = np.random.normal(70, 15, n_samples)   # Weight in kg
kidney_function = np.random.normal(80, 20, n_samples)  # eGFR in mL/min

# Clip values to realistic ranges
age = np.clip(age, 18, 95)
weight = np.clip(weight, 40, 120)
kidney_function = np.clip(kidney_function, 15, 120)

# Drug clearance (L/h) depends on multiple factors
clearance = (5 +  # Baseline clearance
             age_effect * (age - 65) / 10 +  # Age effect
             weight_effect * (weight - 70) / 10 +  # Weight effect
             kidney_effect * (kidney_function - 80) / 10 +  # Kidney function effect
             np.random.normal(0, noise_level, n_samples))  # Random variability

# Create a DataFrame
patient_data = pd.DataFrame({
    'Age': age,
    'Weight': weight,
    'Kidney_Function': kidney_function,
    'Drug_Clearance': clearance
})

# Print the first few rows
print("Patient data:")
print(patient_data.head())

# Calculate correlations
print("\\nCorrelations with Drug Clearance:")
correlations = patient_data.corr()['Drug_Clearance'].sort_values(ascending=False)
print(correlations)

# Perform multiple linear regression
X = patient_data[['Age', 'Weight', 'Kidney_Function']]
y = patient_data['Drug_Clearance']

model = LinearRegression()
model.fit(X, y)

# Get model parameters
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': model.coef_
})

print("\\nMultiple Linear Regression Results:")
print(f"Intercept: {model.intercept_:.3f}")
print("Coefficients:")
print(coefficients)

# Make predictions
y_pred = model.predict(X)
patient_data['Predicted_Clearance'] = y_pred
patient_data['Residual'] = y - y_pred

# Calculate goodness of fit
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - X.shape[1] - 1)

print(f"\\nModel Performance:")
print(f"R² (Coefficient of determination): {r2:.3f}")
print(f"Adjusted R²: {adjusted_r2:.3f}")
print(f"RMSE (Root Mean Square Error): {rmse:.3f} L/h")

# Create regression equation string
equation = f"Clearance = {model.intercept_:.3f}"
for i, var in enumerate(X.columns):
    sign = '+' if model.coef_[i] > 0 else '-'
    equation += f" {sign} {abs(model.coef_[i]):.3f} × {var}"

print(f"\\nRegression Equation:")
print(equation)

# Standardize coefficients to compare relative importance
def standardize(x):
    return (x - x.mean()) / x.std()

X_std = pd.DataFrame()
for col in X.columns:
    X_std[col] = standardize(X[col])

model_std = LinearRegression()
model_std.fit(X_std, y)

std_coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Standardized_Coefficient': model_std.coef_
})

print("\\nStandardized Coefficients (Relative Importance):")
print(std_coefficients.sort_values(by='Standardized_Coefficient', ascending=False))

# Create a visualization of actual vs. predicted clearance
fig = px.scatter(
    patient_data, x='Drug_Clearance', y='Predicted_Clearance',
    title=f"Actual vs. Predicted Drug Clearance (R² = {r2:.3f})",
    labels={
        'Drug_Clearance': 'Actual Clearance (L/h)',
        'Predicted_Clearance': 'Predicted Clearance (L/h)'
    }
)

# Add perfect prediction line
fig.add_trace(
    go.Scatter(
        x=[patient_data['Drug_Clearance'].min(), patient_data['Drug_Clearance'].max()],
        y=[patient_data['Drug_Clearance'].min(), patient_data['Drug_Clearance'].max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    )
)

# Save for display
output_vars['fig'] = fig

# Create bar chart of coefficients for visual comparison
fig2 = px.bar(
    std_coefficients.sort_values(by='Standardized_Coefficient', ascending=False),
    x='Variable', y='Standardized_Coefficient',
    title="Relative Importance of Predictors (Standardized Coefficients)",
    labels={
        'Variable': 'Patient Factor',
        'Standardized_Coefficient': 'Standardized Coefficient'
    },
    color='Standardized_Coefficient',
    color_continuous_scale=px.colors.sequential.Blues,
)

# Save for display
output_vars['fig2'] = fig2

# Create partial dependence plots to visualize the effect of each predictor
reference_age = 65
reference_weight = 70
reference_kidney = 80

fig3 = go.Figure()

# Range of values for each predictor
age_range = np.linspace(20, 90, 50)
weight_range = np.linspace(45, 110, 50)
kidney_range = np.linspace(20, 110, 50)

# Age effect
age_effect_y = model.intercept_ + \
                model.coef_[0] * age_range + \
                model.coef_[1] * reference_weight + \
                model.coef_[2] * reference_kidney

# Weight effect
weight_effect_y = model.intercept_ + \
                   model.coef_[0] * reference_age + \
                   model.coef_[1] * weight_range + \
                   model.coef_[2] * reference_kidney

# Kidney function effect
kidney_effect_y = model.intercept_ + \
                   model.coef_[0] * reference_age + \
                   model.coef_[1] * reference_weight + \
                   model.coef_[2] * kidney_range

# Add traces for each predictor
fig3.add_trace(
    go.Scatter(
        x=age_range, y=age_effect_y,
        mode='lines', name='Age Effect',
        line=dict(color='blue', width=2)
    )
)

fig3.add_trace(
    go.Scatter(
        x=weight_range, y=weight_effect_y,
        mode='lines', name='Weight Effect',
        line=dict(color='green', width=2)
    )
)

fig3.add_trace(
    go.Scatter(
        x=kidney_range, y=kidney_effect_y,
        mode='lines', name='Kidney Function Effect',
        line=dict(color='red', width=2)
    )
)

# Update layout
fig3.update_layout(
    title="Effect of Each Predictor on Drug Clearance",
    xaxis_title="Predictor Value",
    yaxis_title="Predicted Drug Clearance (L/h)",
    height=500
)

# Save for display
output_vars['fig3'] = fig3

# CLINICAL INTERPRETATION

print("\\nClinical Interpretation:")
print("1. Effect of patient factors on drug clearance:")

if abs(model.coef_[0]) > 0.1:
    direction = "decreases" if model.coef_[0] < 0 else "increases"
    print(f"   - Age: Clearance {direction} by {abs(model.coef_[0]):.3f} L/h for each 1-year increase")
    print(f"     (or {abs(model.coef_[0] * 10):.2f} L/h per decade)")
else:
    print("   - Age: Minimal effect on clearance")

if abs(model.coef_[1]) > 0.1:
    direction = "decreases" if model.coef_[1] < 0 else "increases"
    print(f"   - Weight: Clearance {direction} by {abs(model.coef_[1]):.3f} L/h for each 1-kg increase")
    print(f"     (or {abs(model.coef_[1] * 10):.2f} L/h per 10 kg)")
else:
    print("   - Weight: Minimal effect on clearance")

if abs(model.coef_[2]) > 0.1:
    direction = "decreases" if model.coef_[2] < 0 else "increases"
    print(f"   - Kidney Function: Clearance {direction} by {abs(model.coef_[2]):.3f} L/h for each 1-unit increase in eGFR")
    print(f"     (or {abs(model.coef_[2] * 10):.2f} L/h per 10 mL/min)")
else:
    print("   - Kidney Function: Minimal effect on clearance")

print("\\n2. Dosing implications:")

most_important = std_coefficients.sort_values(by='Standardized_Coefficient', key=abs, ascending=False).iloc[0]['Variable']
if most_important == 'Age':
    print("   - Age is the most influential factor - consider age-based dosing")
elif most_important == 'Weight':
    print("   - Weight is the most influential factor - consider weight-based dosing")
else:
    print("   - Kidney function is the most influential factor - consider renal dose adjustments")

if r2 > 0.7:
    print("   - Model has good predictive value - dosing algorithm may be clinically useful")
elif r2 > 0.5:
    print("   - Model has moderate predictive value - use as a general guide for dosing")
else:
    print("   - Model has low predictive value - consider additional factors or therapeutic drug monitoring")

# Calculate the expected clearance for different patient profiles
young_good_kidney = model.intercept_ + model.coef_[0] * 30 + model.coef_[1] * 70 + model.coef_[2] * 100
elderly_poor_kidney = model.intercept_ + model.coef_[0] * 85 + model.coef_[1] * 70 + model.coef_[2] * 30

print("\\n3. Patient examples:")
print(f"   - Young patient with good kidney function (30y, 70kg, eGFR 100): Clearance = {young_good_kidney:.2f} L/h")
print(f"   - Elderly patient with poor kidney function (85y, 70kg, eGFR 30): Clearance = {elderly_poor_kidney:.2f} L/h")
print(f"   - Relative dose adjustment: {elderly_poor_kidney/young_good_kidney:.2f} (or {int(elderly_poor_kidney/young_good_kidney*100)}%)")
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
            - Change the effect parameters (`age_effect`, `weight_effect`, `kidney_effect`) to see how they impact clearance
            - Adjust `noise_level` to see how random variability affects model precision
            - Create different patient populations by adjusting the mean and standard deviation of the characteristics
            
            **Challenges:**
            1. Add a new patient factor (e.g., liver function) that also affects clearance
            2. Create an interaction effect (e.g., age has stronger effect in patients with poor kidney function)
            3. Add code to identify "outlier" patients whose clearance is poorly predicted by the model
            4. Create a visualization that shows which patients need dose adjustments
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
    ### Multiple Linear Regression in Pharmacy:
    
    1. **Model Equation**: Outcome = Intercept + Coefficient₁ × Factor₁ + Coefficient₂ × Factor₂ + ...
        - Each coefficient represents the effect of that factor while holding others constant
        - Allows examination of multiple influences simultaneously
    
    2. **Clinical Applications**:
        - Population pharmacokinetic modeling
        - Personalized dosing algorithms based on multiple patient factors
        - Identifying key determinants of drug response
        - Quantifying factors affecting drug clearance and metabolism
    
    3. **Important Metrics**:
        - R² and Adjusted R²: How well all factors together explain the outcome
        - Standardized coefficients: Relative importance of different factors
        - RMSE: Precision of predictions in the outcome's units
    
    4. **Unique Benefits for Pharmacy Practice**:
        - Isolates independent effects of each patient factor
        - Controls for confounding between related factors (e.g., age and kidney function)
        - Provides basis for complex dosing algorithms
        - Identifies most important factors for therapeutic drug monitoring
    
    5. **Common Predictors in Pharmacy Models**:
        - Demographics: age, weight, sex, race/ethnicity
        - Organ function: renal function (eGFR, CrCl), hepatic function
        - Concomitant medications: drug interactions, inducers, inhibitors
        - Genetic factors: metabolizer status, transporter polymorphisms
    """)

def basic_logistic_regression_lesson():
    st.header("Basic Logistic Regression")
    
    st.markdown("""
    ### Understanding Logistic Regression
    
    Logistic regression models the probability of a binary outcome (yes/no) based on one or more predictor variables.
    
    In pharmacy and epidemiology, logistic regression helps us understand:
    
    - Factors affecting the probability of treatment success
    - Risk of adverse drug reactions
    - Likelihood of medication adherence
    - Probability of disease occurrence
    
    Let's explore basic logistic regression:
    """)
    
    # Initial code example
    initial_code = """# Basic logistic regression
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
n_samples = 200         # Number of patients
dose_effect = 0.8       # Effect of drug dose on response (higher = stronger effect)
age_effect = -0.5       # Effect of age on response (negative = older patients respond less)
add_age_factor = True   # Whether to include age in the model
threshold = 0.5         # Probability threshold for classification

# Generate simulated patient data
np.random.seed(42)  # For reproducible results

# Patient characteristics and treatment
age = np.random.normal(65, 15, n_samples)  # Age in years
age = np.clip(age, 20, 95)  # Realistic age range

dose = np.random.uniform(0, 100, n_samples)  # Drug dose (% of standard dose)

# Generate treatment response (affected by dose and age)
# Using logistic function to model probability
if add_age_factor:
    log_odds = -2 + dose_effect * dose/25 + age_effect * (age - 65)/10
else:
    log_odds = -2 + dose_effect * dose/25  # Only dose affects response
    
response_prob = 1 / (1 + np.exp(-log_odds))
response = np.random.binomial(1, response_prob)  # 1 = success, 0 = failure

# Create DataFrame
patient_data = pd.DataFrame({
    'Age': age,
    'Dose': dose,
    'Response': response,
    'Response_Probability': response_prob
})

# Print the first few rows
print("Patient treatment data:")
print(patient_data.head())

# Basic statistics
print("\\nResponse rate:")
print(f"Overall: {patient_data['Response'].mean():.1%} ({patient_data['Response'].sum()} out of {n_samples})")

# Logistic regression with dose only
X_dose = patient_data[['Dose']]
y = patient_data['Response']

dose_model = LogisticRegression()
dose_model.fit(X_dose, y)

# Get model parameters
dose_coef = dose_model.coef_[0][0]
dose_intercept = dose_model.intercept_[0]
dose_odds_ratio = np.exp(dose_coef)

print("\\nLogistic Regression Results (Dose Only):")
print(f"Intercept: {dose_intercept:.3f}")
print(f"Dose Coefficient: {dose_coef:.3f}")
print(f"Dose Odds Ratio: {dose_odds_ratio:.3f}")

# Interpret odds ratio
print(f"Interpretation: For each 1% increase in dose, the odds of response multiply by {dose_odds_ratio:.3f}")
print(f"For a 10% increase in dose, the odds multiply by {np.exp(dose_coef * 10):.3f}")

# Predictions from dose-only model
dose_probs = dose_model.predict_proba(X_dose)[:, 1]
patient_data['Dose_Model_Probability'] = dose_probs

# Full model with age (if included)
if add_age_factor:
    X_full = patient_data[['Dose', 'Age']]
    full_model = LogisticRegression()
    full_model.fit(X_full, y)
    
    # Get model parameters
    dose_coef_adj = full_model.coef_[0][0]
    age_coef = full_model.coef_[0][1]
    full_intercept = full_model.intercept_[0]
    
    dose_odds_ratio_adj = np.exp(dose_coef_adj)
    age_odds_ratio = np.exp(age_coef)
    
    print("\\nLogistic Regression Results (Full Model):")
    print(f"Intercept: {full_intercept:.3f}")
    print(f"Dose Coefficient: {dose_coef_adj:.3f} (Odds Ratio: {dose_odds_ratio_adj:.3f})")
    print(f"Age Coefficient: {age_coef:.3f} (Odds Ratio: {age_odds_ratio:.3f})")
    
    # Interpret age odds ratio
    if age_coef < 0:
        print(f"Interpretation: For each 1-year increase in age, the odds of response multiply by {age_odds_ratio:.3f}")
        print(f"For a 10-year increase in age, the odds multiply by {np.exp(age_coef * 10):.3f}")
    else:
        print(f"Interpretation: For each 1-year increase in age, the odds of response multiply by {age_odds_ratio:.3f}")
        print(f"For a 10-year increase in age, the odds multiply by {np.exp(age_coef * 10):.3f}")
    
    # Predictions from full model
    full_probs = full_model.predict_proba(X_full)[:, 1]
    patient_data['Full_Model_Probability'] = full_probs

# Model performance (using the full model if available, otherwise dose-only model)
if add_age_factor:
    model_probs = full_probs
    model_name = "Full Model"
else:
    model_probs = dose_probs
    model_name = "Dose-Only Model"

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y, model_probs)
model_auc = auc(fpr, tpr)

print(f"\\n{model_name} Performance:")
print(f"AUC: {model_auc:.3f}")

# Make binary predictions
model_predictions = (model_probs >= threshold).astype(int)

# Classification metrics
print("\\nClassification Report (at threshold {:.1f}):".format(threshold))
report = classification_report(y, model_predictions, output_dict=True)
print(f"Accuracy: {report['accuracy']:.3f}")
print(f"Sensitivity: {report['1']['recall']:.3f}")
print(f"Specificity: {report['0']['recall']:.3f}")
print(f"Positive Predictive Value: {report['1']['precision']:.3f}")

# Confusion matrix
cm = confusion_matrix(y, model_predictions)
print("\\nConfusion Matrix:")
print(cm)

# Create ROC curve plot
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {model_auc:.3f})',
        line=dict(color='blue', width=2)
    )
)

# Add diagonal reference line
fig.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Reference Line',
        line=dict(color='red', dash='dash')
    )
)

# Update layout
fig.update_layout(
    title=f'ROC Curve for {model_name}',
    xaxis_title='False Positive Rate (1 - Specificity)',
    yaxis_title='True Positive Rate (Sensitivity)',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    height=500
)

# Save for display
output_vars['fig'] = fig

# Create a dose-response curve
dose_range = np.linspace(0, 100, 100)

if add_age_factor:
    # For different age groups
    age_groups = [40, 65, 85]
    fig2 = go.Figure()
    
    for age_val in age_groups:
        # Create data for this age
        X_pred = np.column_stack([dose_range, np.full(100, age_val)])
        y_pred_prob = full_model.predict_proba(X_pred)[:, 1]
        
        fig2.add_trace(
            go.Scatter(
                x=dose_range, y=y_pred_prob,
                mode='lines',
                name=f'Age {age_val}',
                line=dict(width=2)
            )
        )
    
    # Add threshold line
    fig2.add_shape(
        type="line",
        x0=0, x1=100,
        y0=threshold, y1=threshold,
        line=dict(color="black", dash="dash"),
    )
    
    # Update layout
    fig2.update_layout(
        title='Dose-Response Probability by Age Group',
        xaxis_title='Dose (% of standard)',
        yaxis_title='Probability of Response',
        height=500
    )
    
else:
    # Dose-only model
    X_pred = dose_range.reshape(-1, 1)
    y_pred_prob = dose_model.predict_proba(X_pred)[:, 1]
    
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Scatter(
            x=dose_range, y=y_pred_prob,
            mode='lines',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add threshold line
    fig2.add_shape(
        type="line",
        x0=0, x1=100,
        y0=threshold, y1=threshold,
        line=dict(color="black", dash="dash"),
    )
    
    # Update layout
    fig2.update_layout(
        title='Dose-Response Probability Curve',
        xaxis_title='Dose (% of standard)',
        yaxis_title='Probability of Response',
        height=500
    )

# Save for display
output_vars['fig2'] = fig2

# Create scatter plot of actual data
fig3 = px.scatter(
    patient_data, x='Dose', y='Age', color='Response',
    title='Treatment Response by Dose and Age',
    labels={'Dose': 'Dose (% of standard)', 'Age': 'Age (years)', 'Response': 'Treatment Response'},
    color_discrete_sequence=['red', 'green']
)

# Save for display
output_vars['fig3'] = fig3

# CLINICAL INTERPRETATION

print("\\nClinical Interpretation:")

# Calculate ED50 (dose for 50% probability of response) for a typical patient
if add_age_factor:
    # For a 65-year-old patient
    # Solve: log_odds = full_intercept + dose_coef_adj * ED50 + age_coef * 65 = 0
    # Where log_odds = 0 corresponds to probability = 0.5
    ED50 = -(full_intercept + age_coef * (65 - 65)/10) / dose_coef_adj * 25
else:
    # Solve: log_odds = dose_intercept + dose_coef * ED50 = 0
    ED50 = -dose_intercept / dose_coef * 25

print(f"ED50 (dose giving 50% probability of response) for a typical 65-year-old patient: {ED50:.1f}%")

# Calculate and interpret odds ratios for clinically relevant changes
if add_age_factor:
    print("\\nEffect of clinical factors on response odds:")
    
    # Dose effect
    dose_change = 25  # 25% increase in dose
    dose_or = np.exp(dose_coef_adj * dose_change/25)
    print(f"- Increasing dose by {dose_change}%: Odds of response multiply by {dose_or:.2f}")
    
    # Age effect
    age_change = 20  # 20 years older
    age_or = np.exp(age_coef * age_change/10)
    print(f"- Patient {age_change} years older: Odds of response multiply by {age_or:.2f}")
    
    # Categorize patients by response probability
    good_responders = (patient_data['Full_Model_Probability'] >= 0.8).mean()
    poor_responders = (patient_data['Full_Model_Probability'] <= 0.2).mean()
    
    print(f"\\nPatient categorization:")
    print(f"- Good responders (≥80% probability): {good_responders:.1%} of patients")
    print(f"- Poor responders (≤20% probability): {poor_responders:.1%} of patients")
    
    # Dose adjustment recommendation
    print("\\nDose adjustment recommendations:")
    print("1. Standard dosing (100%) for average patients")
    
    # Calculate minimum dose needed for 80% response probability in a 65-year-old
    target_log_odds = np.log(0.8 / (1 - 0.8))  # log odds for p = 0.8
    minimum_dose = (target_log_odds - full_intercept - age_coef * (65 - 65)/10) / dose_coef_adj * 25
    minimum_dose = np.clip(minimum_dose, 0, 200)  # Realistic limits
    
    # Calculate minimum dose for 80% response in an 85-year-old
    minimum_dose_elderly = (target_log_odds - full_intercept - age_coef * (85 - 65)/10) / dose_coef_adj * 25
    minimum_dose_elderly = np.clip(minimum_dose_elderly, 0, 200)
    
    print(f"2. Minimum dose for 80% response probability:")
    print(f"   - Average patient (65 years): {minimum_dose:.1f}%")
    print(f"   - Elderly patient (85 years): {minimum_dose_elderly:.1f}%")
    
    if minimum_dose_elderly > 150:
        print("   - Warning: Required dose for elderly patients may exceed safe range")
    
else:
    # Dose-only model interpretations
    print("\\nEffect of dose on response odds:")
    
    # Dose effect
    dose_change = 25  # 25% increase in dose
    dose_or = np.exp(dose_coef * dose_change/25)
    print(f"- Increasing dose by {dose_change}%: Odds of response multiply by {dose_or:.2f}")
    
    # Categorize patients by response probability
    good_responders = (patient_data['Dose_Model_Probability'] >= 0.8).mean()
    poor_responders = (patient_data['Dose_Model_Probability'] <= 0.2).mean()
    
    print(f"\\nPatient categorization:")
    print(f"- Good responders (≥80% probability): {good_responders:.1%} of patients")
    print(f"- Poor responders (≤20% probability): {poor_responders:.1%} of patients")
    
    # Dose adjustment recommendation
    print("\\nDose adjustment recommendations:")
    print("1. Standard dosing (100%) for all patients")
    
    # Calculate minimum dose needed for 80% response probability
    target_log_odds = np.log(0.8 / (1 - 0.8))  # log odds for p = 0.8
    minimum_dose = (target_log_odds - dose_intercept) / dose_coef * 25
    minimum_dose = np.clip(minimum_dose, 0, 200)  # Realistic limits
    
    print(f"2. Minimum dose for 80% response probability: {minimum_dose:.1f}%")
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
            - Change `dose_effect` and `age_effect` to see how they influence response probability
            - Toggle `add_age_factor` to compare simple and multivariable models
            - Adjust `threshold` to see how it affects classification metrics
            
            **Challenges:**
            1. Add a third factor (e.g., comorbidity) that affects treatment response
            2. Create an interaction effect (e.g., dose is less effective in elderly patients)
            3. Add code to calculate the Number Needed to Treat (NNT) for different doses
            4. Find the optimal probability threshold that maximizes sensitivity + specificity
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
    ### Logistic Regression in Pharmacy:

    1. **Model Equation**: log(p/(1-p)) = Intercept + Coefficient₁ × Factor₁ + ...
       - Models the log odds of a binary outcome
       - Produces an S-shaped (sigmoid) probability curve
       - Coefficients show the change in log odds per unit change in the predictor

    2. **Clinical Applications**:
       - Predicting treatment response probability
       - Identifying patients at risk for adverse events
       - Developing clinical decision rules
       - Risk stratification for medication management
       - Determining factors affecting medication adherence

    3. **Important Metrics**:
       - **Odds Ratio**: Exponentiated coefficient, shows how odds multiply per unit change
       - **AUC**: Area Under the ROC Curve, measures discrimination ability (0.5-1.0)
       - **Sensitivity**: Proportion of true positives correctly identified
       - **Specificity**: Proportion of true negatives correctly identified

    4. **Unique Benefits for Pharmacy Practice**:
       - Enables risk-based approaches to medication management
       - Helps identify which patients need dose modifications
       - Allows calculation of ED50 and other probability thresholds
       - Supports individualized benefit-risk assessment

    5. **Interpreting Odds Ratios**:
       - OR > 1: Factor increases the odds of the outcome
       - OR < 1: Factor decreases the odds of the outcome
       - OR = 2: Odds double with each unit increase in the factor
       - OR = 0.5: Odds halve with each unit increase in the factor
    """)

def interpreting_regression_lesson():
    st.header("Interpreting Regression Models")
    
    st.markdown("""
    ### Understanding Regression Output and Interpretation
    
    Interpreting regression results correctly is essential for applying them in clinical practice.
    This lesson focuses on how to extract meaningful insights from regression output and communicate
    them effectively.
    
    Let's explore how to interpret common regression metrics and coefficients:
    """)
    
    # Initial code example
    initial_code = """# Interpreting regression models
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, auc, confusion_matrix
from scipy import stats

# Load simulated clinical data
np.random.seed(42)  # For reproducible results

# Create a function to generate simulated therapeutic data
def generate_clinical_data(n_samples=200):
    # Patient characteristics
    age = np.random.normal(60, 15, n_samples)
    weight = np.random.normal(75, 15, n_samples)
    female = np.random.binomial(1, 0.5, n_samples)
    
    # Clip to realistic values
    age = np.clip(age, 18, 95)
    weight = np.clip(weight, 45, 150)
    
    # Create some correlation between variables (age and weight)
    weight = weight - 0.3 * (age - 60) + np.random.normal(0, 10, n_samples)
    
    # Set up coefficients for two outcomes
    # 1. Drug concentration (continuous outcome)
    conc_intercept = 50
    conc_age_effect = -0.3
    conc_weight_effect = 0.25
    conc_female_effect = -15
    
    # 2. Efficacy (binary outcome)
    eff_intercept = -2
    eff_age_effect = -0.03
    eff_weight_effect = 0.01
    eff_female_effect = 0.5
    eff_conc_effect = 0.05
    
    # Generate drug concentration (μg/L)
    concentration = (conc_intercept + 
                    conc_age_effect * (age - 60) + 
                    conc_weight_effect * (weight - 75) + 
                    conc_female_effect * female + 
                    np.random.normal(0, 10, n_samples))
    
    # Generate efficacy (binary outcome)
    efficacy_logodds = (eff_intercept + 
                       eff_age_effect * (age - 60) + 
                       eff_weight_effect * (weight - 75) + 
                       eff_female_effect * female +
                       eff_conc_effect * (concentration - 50) + 
                       np.random.normal(0, 1, n_samples))
    
    efficacy_prob = 1 / (1 + np.exp(-efficacy_logodds))
    efficacy = np.random.binomial(1, efficacy_prob)
    
    # Create DataFrame
    return pd.DataFrame({
        'Age': age,
        'Weight': weight,
        'Female': female,
        'Concentration': concentration,
        'Efficacy': efficacy,
        'Efficacy_Probability': efficacy_prob
    })

# Generate the data
clinical_data = generate_clinical_data(200)

# Print dataset summary
print("Clinical Dataset Summary:")
print(clinical_data.describe())

# Print correlation matrix
print("\\nCorrelation Matrix:")
print(clinical_data.corr().round(2))

print("\\n" + "="*50)
print("PART 1: INTERPRETING LINEAR REGRESSION")
print("="*50)

# Fit linear regression model for drug concentration
X_conc = clinical_data[['Age', 'Weight', 'Female']]
y_conc = clinical_data['Concentration']

lin_model = LinearRegression()
lin_model.fit(X_conc, y_conc)

# Make predictions
y_conc_pred = lin_model.predict(X_conc)
clinical_data['Predicted_Concentration'] = y_conc_pred

# Calculate performance metrics
r2 = r2_score(y_conc, y_conc_pred)
rmse = np.sqrt(mean_squared_error(y_conc, y_conc_pred))

# Create coefficient dataframe with confidence intervals
def calculate_linear_regression_stats(X, y, model):
    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Get degrees of freedom and MSE
    n = len(y)
    p = X.shape[1]
    df = n - p - 1
    mse = np.sum(residuals**2) / df
    
    # Calculate standard errors
    X_with_const = np.column_stack([np.ones(n), X])
    cov_matrix = mse * np.linalg.inv(X_with_const.T.dot(X_with_const))
    se = np.sqrt(np.diag(cov_matrix))[1:]  # Skip intercept SE
    
    # Calculate t-statistics and p-values
    t_stats = model.coef_ / se
    p_values = [2 * (1 - stats.t.cdf(abs(t), df)) for t in t_stats]
    
    # Calculate confidence intervals
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = model.coef_ - t_critical * se
    ci_upper = model.coef_ + t_critical * se
    
    # Create results dataframe
    results = pd.DataFrame({
        'Coefficient': model.coef_,
        'Standard_Error': se,
        'T_Statistic': t_stats,
        'P_Value': p_values,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    }, index=X.columns)
    
    # Add intercept to results
    intercept_se = np.sqrt(cov_matrix[0, 0])
    intercept_t = model.intercept_ / intercept_se
    intercept_p = 2 * (1 - stats.t.ppf(abs(intercept_t), df))
    intercept_ci_lower = model.intercept_ - t_critical * intercept_se
    intercept_ci_upper = model.intercept_ + t_critical * intercept_se
    
    intercept_row = pd.DataFrame({
        'Coefficient': [model.intercept_],
        'Standard_Error': [intercept_se],
        'T_Statistic': [intercept_t],
        'P_Value': [intercept_p],
        'CI_Lower': [intercept_ci_lower],
        'CI_Upper': [intercept_ci_upper]
    }, index=['Intercept'])
    
    return pd.concat([intercept_row, results])

# Get detailed linear regression statistics
lin_results = calculate_linear_regression_stats(X_conc, y_conc, lin_model)

# Print model summary
print("\\nLinear Regression Model for Drug Concentration:")
print(f"R² = {r2:.3f}, RMSE = {rmse:.3f} μg/L")
print("\\nCoefficients with 95% Confidence Intervals:")
print(lin_results.round(3))

# Interpret linear regression results
print("\\nInterpretation of Linear Regression Results:")
print("1. Intercept: Estimated concentration for a 60-year-old male weighing 75 kg")
print(f"   Value: {lin_results.loc['Intercept', 'Coefficient']:.2f} μg/L (95% CI: {lin_results.loc['Intercept', 'CI_Lower']:.2f} to {lin_results.loc['Intercept', 'CI_Upper']:.2f})")

print("\\n2. Age Effect: Change in concentration for each year increase in age")
coef = lin_results.loc['Age', 'Coefficient']
ci_lower = lin_results.loc['Age', 'CI_Lower']
ci_upper = lin_results.loc['Age', 'CI_Upper']
p_value = lin_results.loc['Age', 'P_Value']
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

print(f"   Value: {coef:.2f} μg/L per year (95% CI: {ci_lower:.2f} to {ci_upper:.2f})")
print(f"   Interpretation: For each additional year of age, concentration {abs(coef):.2f} μg/L {'decreases' if coef < 0 else 'increases'}")
print(f"   Clinical significance: A 10-year age difference corresponds to a {abs(coef * 10):.1f} μg/L {'decrease' if coef < 0 else 'increase'} in concentration")
print(f"   Statistical significance: This effect is {significance} (p = {p_value:.4f})")

print("\\n3. Weight Effect: Change in concentration for each kg increase in weight")
coef = lin_results.loc['Weight', 'Coefficient']
ci_lower = lin_results.loc['Weight', 'CI_Lower']
ci_upper = lin_results.loc['Weight', 'CI_Upper']
p_value = lin_results.loc['Weight', 'P_Value']
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

print(f"   Value: {coef:.2f} μg/L per kg (95% CI: {ci_lower:.2f} to {ci_upper:.2f})")
print(f"   Interpretation: For each additional kg of weight, concentration {abs(coef):.2f} μg/L {'decreases' if coef < 0 else 'increases'}")
print(f"   Clinical significance: A 10 kg weight difference corresponds to a {abs(coef * 10):.1f} μg/L {'decrease' if coef < 0 else 'increase'} in concentration")
print(f"   Statistical significance: This effect is {significance} (p = {p_value:.4f})")

print("\\n4. Sex Effect: Difference in concentration for females compared to males")
coef = lin_results.loc['Female', 'Coefficient']
ci_lower = lin_results.loc['Female', 'CI_Lower']
ci_upper = lin_results.loc['Female', 'CI_Upper']
p_value = lin_results.loc['Female', 'P_Value']
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

print(f"   Value: {coef:.2f} μg/L (95% CI: {ci_lower:.2f} to {ci_upper:.2f})")
print(f"   Interpretation: Females have {abs(coef):.2f} μg/L {'lower' if coef < 0 else 'higher'} concentrations than males, on average")
print(f"   Statistical significance: This effect is {significance} (p = {p_value:.4f})")

print("\\nOverall Model Assessment:")
print(f"1. R² = {r2:.3f}: The model explains {r2*100:.1f}% of the variability in drug concentration")
print(f"2. RMSE = {rmse:.3f} μg/L: The typical error in predicting concentration is about {rmse:.1f} μg/L")

if r2 > 0.7:
    print("3. The model has good predictive power (R² > 0.7)")
elif r2 > 0.5:
    print("3. The model has moderate predictive power (R² > 0.5)")
else:
    print("3. The model has limited predictive power (R² < 0.5)")

print("\\nClinical Application Example:")
example_patient = pd.DataFrame({
    'Age': [75],
    'Weight': [60],
    'Female': [1]
})

predicted_conc = lin_model.predict(example_patient)[0]
print(f"For a 75-year-old female weighing 60 kg:")
print(f"Predicted concentration: {predicted_conc:.1f} μg/L")

standard_patient = pd.DataFrame({
    'Age': [60],
    'Weight': [75],
    'Female': [0]
})
standard_conc = lin_model.predict(standard_patient)[0]
percent_diff = (predicted_conc / standard_conc - 1) * 100

print(f"Compared to a standard 60-year-old male (75 kg) with concentration {standard_conc:.1f} μg/L:")
print(f"This represents a {percent_diff:.1f}% {'decrease' if percent_diff < 0 else 'increase'}")

# Create visualization of coefficients with confidence intervals
fig = go.Figure()

# Plot coefficients and CIs
coef_names = ['Age Effect', 'Weight Effect', 'Female Effect']
coef_values = lin_results.loc[['Age', 'Weight', 'Female'], 'Coefficient'].values
ci_lower = lin_results.loc[['Age', 'Weight', 'Female'], 'CI_Lower'].values
ci_upper = lin_results.loc[['Age', 'Weight', 'Female'], 'CI_Upper'].values

# Add coefficient points
fig.add_trace(go.Scatter(
    x=coef_values,
    y=coef_names,
    mode='markers',
    marker=dict(size=10, color='blue'),
    name='Coefficient'
))

# Add error bars for CIs
fig.add_trace(go.Scatter(
    x=ci_upper,
    y=coef_names,
    mode='markers',
    marker=dict(size=8, color='blue', symbol='line-ns-open'),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=ci_lower,
    y=coef_names,
    mode='markers',
    marker=dict(size=8, color='blue', symbol='line-ns-open'),
    showlegend=False
))

# Add lines for CIs
for i, name in enumerate(coef_names):
    fig.add_shape(
        type="line",
        x0=ci_lower[i], x1=ci_upper[i],
        y0=name, y1=name,
        line=dict(color="blue", width=2)
    )

# Add vertical line at zero
fig.add_shape(
    type="line",
    x0=0, x1=0,
    y0=-0.5, y1=len(coef_names) - 0.5,
    line=dict(color="red", width=1, dash="dash")
)

# Update layout
fig.update_layout(
    title="Linear Regression Coefficients with 95% Confidence Intervals",
    xaxis_title="Coefficient Value (Effect on Concentration in μg/L)",
    yaxis_title="Patient Factor",
    height=400
)

# Save for display
output_vars['fig'] = fig

print("\\n" + "="*50)
print("PART 2: INTERPRETING LOGISTIC REGRESSION")
print("="*50)

# Fit logistic regression model for efficacy
X_eff = clinical_data[['Age', 'Weight', 'Female', 'Concentration']]
y_eff = clinical_data['Efficacy']

log_model = LogisticRegression()
log_model.fit(X_eff, y_eff)

# Make predictions
y_eff_pred_prob = log_model.predict_proba(X_eff)[:, 1]
clinical_data['Predicted_Efficacy_Prob'] = y_eff_pred_prob
y_eff_pred = (y_eff_pred_prob >= 0.5).astype(int)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_eff, y_eff_pred_prob)
model_auc = auc(fpr, tpr)

# Calculate confusion matrix
cm = confusion_matrix(y_eff, y_eff_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate accuracy, sensitivity, specificity
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

# Calculate odds ratios and confidence intervals for logistic regression
def calculate_logistic_regression_stats(X, y, model):
    # Get number of samples and features
    n, p = X.shape
    
    # Get predictions and calculate variance-covariance matrix
    y_pred_prob = model.predict_proba(X)[:, 1]
    
    # Calculate variance-covariance matrix (simplified approximation)
    X_with_const = np.column_stack([np.ones(n), X])
    W = np.diag(y_pred_prob * (1 - y_pred_prob))
    cov_matrix = np.linalg.inv(X_with_const.T.dot(W).dot(X_with_const))
    
    # Standard errors for coefficients
    se = np.sqrt(np.diag(cov_matrix))[1:]  # Skip intercept SE
    
    # Calculate z-statistics and p-values
    z_stats = model.coef_[0] / se
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_stats]
    
# Calculate confidence intervals for log odds
    ci_lower_log = model.coef_[0] - 1.96 * se
    ci_upper_log = model.coef_[0] + 1.96 * se
    
    # Convert to odds ratios and CI
    odds_ratios = np.exp(model.coef_[0])
    ci_lower_or = np.exp(ci_lower_log)
    ci_upper_or = np.exp(ci_upper_log)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Coefficient': model.coef_[0],
        'Standard_Error': se,
        'Z_Statistic': z_stats,
        'P_Value': p_values,
        'Odds_Ratio': odds_ratios,
        'OR_CI_Lower': ci_lower_or,
        'OR_CI_Upper': ci_upper_or
    }, index=X.columns)
    
    # Add intercept info
    intercept_se = np.sqrt(cov_matrix[0, 0])
    intercept_z = model.intercept_[0] / intercept_se
    intercept_p = 2 * (1 - stats.norm.cdf(abs(intercept_z)))
    
    intercept_ci_lower = model.intercept_[0] - 1.96 * intercept_se
    intercept_ci_upper = model.intercept_[0] + 1.96 * intercept_se
    
    # No odds ratio for intercept, but we'll add the log odds info
    intercept_row = pd.DataFrame({
        'Coefficient': [model.intercept_[0]],
        'Standard_Error': [intercept_se],
        'Z_Statistic': [intercept_z],
        'P_Value': [intercept_p],
        'Odds_Ratio': [np.exp(model.intercept_[0])],
        'OR_CI_Lower': [np.exp(intercept_ci_lower)],
        'OR_CI_Upper': [np.exp(intercept_ci_upper)]
    }, index=['Intercept'])
    
    return pd.concat([intercept_row, results])

# Get detailed logistic regression statistics
log_results = calculate_logistic_regression_stats(X_eff, y_eff, log_model)

# Print model summary
print("\nLogistic Regression Model for Treatment Efficacy:")
print(f"AUC = {model_auc:.3f}, Accuracy = {accuracy:.3f}")
print(f"Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}")
print("\nOdds Ratios with 95% Confidence Intervals:")
print(log_results[['Odds_Ratio', 'OR_CI_Lower', 'OR_CI_Upper', 'P_Value']].round(3))

# Interpret logistic regression results
print("\nInterpretation of Logistic Regression Results:")
print("1. Intercept: Log odds of efficacy for a 60-year-old male weighing 75 kg with concentration of 50 μg/L")
print(f"   Log odds: {log_results.loc['Intercept', 'Coefficient']:.2f}")
print(f"   Baseline probability: {1/(1+np.exp(-log_results.loc['Intercept', 'Coefficient'])):.2f}")

print("\n2. Age Effect: Change in odds of efficacy for each year increase in age")
coef = log_results.loc['Age', 'Coefficient']
or_val = log_results.loc['Age', 'Odds_Ratio']
ci_lower = log_results.loc['Age', 'OR_CI_Lower']
ci_upper = log_results.loc['Age', 'OR_CI_Upper']
p_value = log_results.loc['Age', 'P_Value']
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

print(f"   Odds Ratio: {or_val:.3f} (95% CI: {ci_lower:.3f} to {ci_upper:.3f})")
print(f"   Interpretation: For each additional year of age, the odds of efficacy multiply by {or_val:.3f}")
print(f"   Clinical significance: A 10-year age difference corresponds to an odds ratio of {or_val**10:.3f}")
print(f"   Statistical significance: This effect is {significance} (p = {p_value:.4f})")

print("\n3. Weight Effect: Change in odds of efficacy for each kg increase in weight")
coef = log_results.loc['Weight', 'Coefficient']
or_val = log_results.loc['Weight', 'Odds_Ratio']
ci_lower = log_results.loc['Weight', 'OR_CI_Lower']
ci_upper = log_results.loc['Weight', 'OR_CI_Upper']
p_value = log_results.loc['Weight', 'P_Value']
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

print(f"   Odds Ratio: {or_val:.3f} (95% CI: {ci_lower:.3f} to {ci_upper:.3f})")
print(f"   Interpretation: For each additional kg of weight, the odds of efficacy multiply by {or_val:.3f}")
print(f"   Clinical significance: A 10 kg weight difference corresponds to an odds ratio of {or_val**10:.3f}")
print(f"   Statistical significance: This effect is {significance} (p = {p_value:.4f})")

print("\n4. Sex Effect: Difference in odds of efficacy for females compared to males")
coef = log_results.loc['Female', 'Coefficient']
or_val = log_results.loc['Female', 'Odds_Ratio']
ci_lower = log_results.loc['Female', 'OR_CI_Lower']
ci_upper = log_results.loc['Female', 'OR_CI_Upper']
p_value = log_results.loc['Female', 'P_Value']
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

print(f"   Odds Ratio: {or_val:.3f} (95% CI: {ci_lower:.3f} to {ci_upper:.3f})")
direction = "higher" if or_val > 1 else "lower"
print(f"   Interpretation: Females have {or_val:.2f} times the odds of efficacy compared to males")
print(f"   Statistical significance: This effect is {significance} (p = {p_value:.4f})")

print("\n5. Concentration Effect: Change in odds of efficacy for each μg/L increase in concentration")
coef = log_results.loc['Concentration', 'Coefficient']
or_val = log_results.loc['Concentration', 'Odds_Ratio']
ci_lower = log_results.loc['Concentration', 'OR_CI_Lower']
ci_upper = log_results.loc['Concentration', 'OR_CI_Upper']
p_value = log_results.loc['Concentration', 'P_Value']
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

print(f"   Odds Ratio: {or_val:.3f} (95% CI: {ci_lower:.3f} to {ci_upper:.3f})")
print(f"   Interpretation: For each additional μg/L in concentration, the odds of efficacy multiply by {or_val:.3f}")
print(f"   Clinical significance: A 10 μg/L increase in concentration corresponds to an odds ratio of {or_val**10:.3f}")
print(f"   Statistical significance: This effect is {significance} (p = {p_value:.4f})")

print("\nOverall Model Assessment:")
print(f"1. AUC = {model_auc:.3f}: The model has {'excellent' if model_auc > 0.9 else 'good' if model_auc > 0.8 else 'fair' if model_auc > 0.7 else 'poor'} discrimination")
print(f"2. Classification metrics at 0.5 threshold:")
print(f"   - Accuracy: {accuracy:.3f} (overall correct classification)")
print(f"   - Sensitivity: {sensitivity:.3f} (correctly identified efficacy)")
print(f"   - Specificity: {specificity:.3f} (correctly identified non-efficacy)")
print(f"   - Positive Predictive Value: {ppv:.3f} (probability of efficacy given positive prediction)")

print("\nClinical Application Example:")
example_patient = pd.DataFrame({
    'Age': [75],
    'Weight': [60],
    'Female': [1],
    'Concentration': [40]
})

predicted_prob = log_model.predict_proba(example_patient)[0, 1]
print(f"For a 75-year-old female weighing 60 kg with concentration of 40 μg/L:")
print(f"Predicted probability of efficacy: {predicted_prob:.3f} or {predicted_prob*100:.1f}%")

standard_patient = pd.DataFrame({
    'Age': [60],
    'Weight': [75],
    'Female': [0],
    'Concentration': [50]
})
standard_prob = log_model.predict_proba(standard_patient)[0, 1]

print(f"Compared to a standard 60-year-old male (75 kg) with concentration 50 μg/L and efficacy probability of {standard_prob:.3f}:")
relative_risk = predicted_prob / standard_prob
print(f"Relative probability: {relative_risk:.2f}")

# Create a plot of odds ratios with confidence intervals
fig2 = go.Figure()

# Get the data for the plot
factors = ['Age', 'Weight', 'Female', 'Concentration']
odds_ratios = log_results.loc[factors, 'Odds_Ratio'].values
ci_lower = log_results.loc[factors, 'OR_CI_Lower'].values
ci_upper = log_results.loc[factors, 'OR_CI_Upper'].values

# Create axis labels
axis_labels = [
    f'Age (per year)',
    f'Weight (per kg)',
    f'Female (vs Male)',
    f'Concentration (per μg/L)'
]

# Plot odds ratios on a log scale
fig2.add_trace(go.Scatter(
    x=odds_ratios,
    y=axis_labels,
    mode='markers',
    marker=dict(size=10, color='blue'),
    name='Odds Ratio'
))

# Add error bars for CIs
for i, label in enumerate(axis_labels):
    fig2.add_shape(
        type="line",
        x0=ci_lower[i], x1=ci_upper[i],
        y0=label, y1=label,
        line=dict(color="blue", width=2)
    )
    
    # Add markers for CI bounds
    fig2.add_trace(go.Scatter(
        x=[ci_lower[i], ci_upper[i]],
        y=[label, label],
        mode='markers',
        marker=dict(size=8, color='blue', symbol='line-ns-open'),
        showlegend=False
    ))

# Add vertical line at OR=1 (no effect)
fig2.add_shape(
    type="line",
    x0=1, x1=1,
    y0=-0.5, y1=len(factors) - 0.5,
    line=dict(color="red", width=1, dash="dash")
)

# Use log scale for x-axis
min_val = min(ci_lower) * 0.8
max_val = max(ci_upper) * 1.2

fig2.update_layout(
    title="Odds Ratios with 95% Confidence Intervals",
    xaxis_title="Odds Ratio (log scale)",
    xaxis_type="log",
    xaxis=dict(range=[np.log10(min_val), np.log10(max_val)]),
    height=400
)

# Save for display
output_vars['fig2'] = fig2

# Create ROC curve
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name=f'ROC Curve (AUC = {model_auc:.3f})',
    line=dict(color='blue', width=2)
))

# Add diagonal reference line
fig3.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Reference',
    line=dict(color='red', dash='dash')
))

# Update layout
fig3.update_layout(
    title='ROC Curve for Efficacy Model',
    xaxis_title='False Positive Rate (1 - Specificity)',
    yaxis_title='True Positive Rate (Sensitivity)',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    height=500
)

# Save for display
output_vars['fig3'] = fig3

# Create predicted probability by concentration curve for different patient groups
fig4 = go.Figure()

# Create concentration range
conc_range = np.linspace(20, 80, 100)

# Create different patient profiles
patient_profiles = [
    {'name': 'Young Male', 'Age': 40, 'Weight': 75, 'Female': 0},
    {'name': 'Young Female', 'Age': 40, 'Weight': 75, 'Female': 1},
    {'name': 'Elderly Male', 'Age': 80, 'Weight': 75, 'Female': 0},
    {'name': 'Elderly Female', 'Age': 80, 'Weight': 75, 'Female': 1}
]

for profile in patient_profiles:
    # Create test data
    test_data = pd.DataFrame({
        'Age': [profile['Age']] * 100,
        'Weight': [profile['Weight']] * 100,
        'Female': [profile['Female']] * 100,
        'Concentration': conc_range
    })
    
    # Get predictions
    pred_prob = log_model.predict_proba(test_data)[:, 1]
    
    # Add trace
    fig4.add_trace(go.Scatter(
        x=conc_range,
        y=pred_prob,
        mode='lines',
        name=profile['name'],
        line=dict(width=2)
    ))

# Add horizontal line at p=0.5
fig4.add_shape(
    type="line",
    x0=20, x1=80,
    y0=0.5, y1=0.5,
    line=dict(color="black", dash="dash")
)

fig4.update_layout(
    title='Predicted Probability of Efficacy by Concentration',
    xaxis_title='Drug Concentration (μg/L)',
    yaxis_title='Probability of Efficacy',
    height=500
)

# Save for display
output_vars['fig4'] = fig4
"""

    # Create an editable text area with the initial code
    user_code = st.text_area("Modify the code and click Execute to see the results:", 
                           value=initial_code, height=400)
    
    # Execute button and handling
    execute_button = st.button("Run Code")
    
    if execute_button:
        success, output, output_vars = execute_code(user_code)
        
        if success:
            if output:
                st.subheader("Output:")
                st.text(output)
            
            if 'fig' in output_vars:
                st.plotly_chart(output_vars['fig'], use_container_width=True)
            
            if 'fig2' in output_vars:
                st.plotly_chart(output_vars['fig2'], use_container_width=True)
        else:
            st.error("Error in your code:")
            st.code(output)