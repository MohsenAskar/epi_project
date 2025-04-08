import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, auc
import sys
import os

# Add path to import the code lab module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interactive_code'))
import regression_code

st.set_page_config(layout="wide")
# Title Section
st.title("Understanding Regression Analysis in Epidemiology")

# Create tabs to separate visualization from code lab
viz_tab, code_tab = st.tabs(["📊 Interactive Visualization", "💻 Code Laboratory"])

# Visualization tab content
with viz_tab:
    st.header("Introduction to Regression Analysis")
    
    st.markdown("""
    **Regression analysis** is a statistical method used to examine relationships between variables. 
    In epidemiology and pharmacy, it helps us understand:
    
    - How medication dosage relates to patient outcomes
    - Relationships between risk factors and disease probability
    - Factors influencing medication adherence
    - Dose-response relationships for drugs
    
    Let's explore the two main types of regression:
    """)
    
    regression_type = st.radio(
        "Select regression type to explore:",
        ["Linear Regression", "Logistic Regression"]
    )
    
    if regression_type == "Linear Regression":
        st.subheader("Linear Regression")
        
        st.markdown("""
        **Linear regression** models the relationship between a dependent variable (y) and one or more 
        independent variables (x) using a linear equation:
        
        y = β₀ + β₁x₁ + β₂x₂ + ... + ε
        
        Where:
        - y is the outcome or dependent variable
        - x₁, x₂, etc. are predictor or independent variables
        - β₀ is the y-intercept (value when all x = 0)
        - β₁, β₂, etc. are the slopes or coefficients
        - ε is the error term
        
        The **fit line** (or regression line) is the line that minimizes the sum of squared differences 
        between observed values and predicted values.
        """)
        
        # Create columns for controls and visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Adjust Parameters")
            
            # Linear regression parameters
            sample_size = st.slider(
                "Sample Size",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
            
            correlation = st.slider(
                "Relationship Strength",
                min_value=-1.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
            
            noise = st.slider(
                "Noise Level",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
            
            show_residuals = st.checkbox("Show Residuals", value=False)
            
            dataset_choice = st.selectbox(
                "Sample Dataset",
                ["Drug Concentration vs Effect", "Medication Adherence", "Blood Pressure Reduction"]
            )
        
        with col2:
            # Generate data based on parameters and dataset choice
            np.random.seed(42)  # For reproducibility
            
            if dataset_choice == "Drug Concentration vs Effect":
                x_label = "Drug Concentration (mg/L)"
                y_label = "Therapeutic Effect (%)"
                x = np.random.uniform(0, 10, sample_size)
                y = correlation * x + np.random.normal(0, noise, sample_size)
                # Normalize therapeutic effect to be between 0 and 100%
                y = 50 + 10 * y
                y = np.clip(y, 0, 100)
                
            elif dataset_choice == "Medication Adherence":
                x_label = "Medication Adherence Score"
                y_label = "Treatment Success Rate (%)"
                x = np.random.uniform(0, 10, sample_size)
                y = correlation * x + np.random.normal(0, noise, sample_size)
                # Scale to appropriate range
                y = 40 + 6 * y
                y = np.clip(y, 0, 100)
                
            else:  # Blood Pressure Reduction
                x_label = "Medication Dose (mg)"
                y_label = "Blood Pressure Reduction (mmHg)"
                x = np.random.uniform(50, 200, sample_size)
                y = correlation * (x - 50) / 30 + np.random.normal(0, noise, sample_size)
                # Scale to appropriate range
                y = 5 + 2 * y
                y = np.clip(y, 0, 30)
            
            # Fit linear regression
            X = x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Create DataFrame for plotting
            data = pd.DataFrame({
                'x': x,
                'y': y,
                'y_pred': y_pred,
                'residuals': residuals
            })
            
            # Plot scatter and regression line
            fig = px.scatter(
                data, x='x', y='y',
                labels={'x': x_label, 'y': y_label},
                title=f"Linear Regression (R² = {r2:.3f}, RMSE = {rmse:.3f})"
            )
            
            # Add regression line
            x_range = np.linspace(min(x), max(x), 100)
            y_range = model.predict(x_range.reshape(-1, 1))
            
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=y_range,
                    mode='lines',
                    name='Regression Line',
                    line=dict(color='red', width=2)
                )
            )
            
            if show_residuals:
                # Add residual lines
                for i in range(len(x)):
                    fig.add_shape(
                        type="line",
                        x0=x[i], y0=y[i],
                        x1=x[i], y1=y_pred[i],
                        line=dict(color="green", width=1, dash="dash"),
                    )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display model summary
            st.subheader("Model Summary")
            st.write(f"Equation: {y_label} = {model.intercept_:.3f} + {model.coef_[0]:.3f} × {x_label}")
            st.write(f"R² (Coefficient of Determination): {r2:.3f}")
            st.write(f"RMSE (Root Mean Square Error): {rmse:.3f}")
            
            # Add residual plot if enabled
            if show_residuals:
                fig_resid = px.scatter(
                    data, x='y_pred', y='residuals',
                    labels={'y_pred': f'Predicted {y_label}', 'residuals': 'Residuals'},
                    title="Residual Plot"
                )
                
                fig_resid.add_shape(
                    type="line",
                    x0=min(y_pred), y0=0,
                    x1=max(y_pred), y1=0,
                    line=dict(color="red", width=2, dash="dash"),
                )
                
                st.plotly_chart(fig_resid, use_container_width=True)
        
        # Interpretation section
        st.subheader("Interpreting Linear Regression Results")
        
        st.markdown(f"""
        ### Interpretation for {dataset_choice}:
        
        1. **The slope coefficient ({model.coef_[0]:.3f})** represents:
           - The change in {y_label} for each one-unit increase in {x_label}
           - A positive value means as {x_label} increases, {y_label} tends to increase
           - A negative value would mean as {x_label} increases, {y_label} tends to decrease
        
        2. **The intercept ({model.intercept_:.3f})** represents:
           - The predicted {y_label} when {x_label} is zero
           - Note: Sometimes the intercept doesn't have a meaningful interpretation, 
             especially if zero is outside the range of observed {x_label} values
        
        3. **R² value ({r2:.3f})** tells us:
           - The proportion of variance in {y_label} explained by {x_label}
           - R² ranges from 0 to 1, with higher values indicating a better fit
           - This model explains {r2*100:.1f}% of the variation in {y_label}
        
        4. **RMSE ({rmse:.3f})** represents:
           - The typical difference between predicted and actual values
           - Lower values indicate better prediction accuracy
           - RMSE is in the same units as {y_label}
        """)
        
        st.subheader("Applications in Pharmacy")
        
        application_examples = {
            "Drug Concentration vs Effect": """
            - Modeling drug concentration-effect relationships
            - Understanding therapeutic window boundaries
            - Determining optimal dosing for maximum efficacy
            - Predicting therapeutic outcomes based on measured drug levels
            """,
            "Medication Adherence": """
            - Quantifying the impact of adherence on treatment outcomes
            - Predicting success rates based on patient adherence scores
            - Evaluating interventions designed to improve adherence
            - Identifying the minimum adherence threshold for therapeutic benefit
            """,
            "Blood Pressure Reduction": """
            - Establishing dose-response curves for antihypertensive medications
            - Predicting blood pressure reduction at specific doses
            - Comparing efficacy between different antihypertensive agents
            - Identifying optimal dosing for specific patient populations
            """
        }
        
        st.markdown(application_examples[dataset_choice])
        
    else:  # Logistic Regression
        st.subheader("Logistic Regression")
        
        st.markdown("""
        **Logistic regression** models the probability of a binary outcome based on one or more predictor variables:
        
        log(p/(1-p)) = β₀ + β₁x₁ + β₂x₂ + ... + ε
        
        Where:
        - p is the probability of the outcome occurring
        - log(p/(1-p)) is the log odds (logit) of the outcome
        - x₁, x₂, etc. are predictor variables
        - β₀, β₁, β₂, etc. are the coefficients
        
        The **fit curve** is S-shaped (sigmoid) and represents the probability of the outcome 
        at different values of the predictor.
        """)
        
        # Create columns for controls and visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Adjust Parameters")
            
            # Logistic regression parameters
            sample_size = st.slider(
                "Sample Size",
                min_value=20,
                max_value=300,
                value=100,
                step=20
            )
            
            effect_strength = st.slider(
                "Effect Strength",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1
            )
            
            intercept = st.slider(
                "Intercept (Baseline Risk)",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.2
            )
            
            dataset_choice = st.selectbox(
                "Sample Dataset",
                ["Drug Response", "Adverse Event Risk", "Treatment Success"]
            )
            
            show_probabilities = st.checkbox("Show Individual Probabilities", value=False)
        
        with col2:
            # Generate data based on parameters and dataset choice
            np.random.seed(42)  # For reproducibility
            
            if dataset_choice == "Drug Response":
                x_label = "Drug Plasma Concentration (mg/L)"
                y_label = "Response (1=Yes, 0=No)"
                x = np.random.uniform(0, 10, sample_size)
                
            elif dataset_choice == "Adverse Event Risk":
                x_label = "Patient Age (years)"
                y_label = "Adverse Event (1=Yes, 0=No)"
                x = np.random.normal(65, 15, sample_size)
                x = np.clip(x, 20, 100)
                
            else:  # Treatment Success
                x_label = "Medication Adherence (%)"
                y_label = "Treatment Success (1=Yes, 0=No)"
                x = np.random.uniform(20, 100, sample_size)
            
            # Generate probabilities based on logistic function
            logit = intercept + effect_strength * x
            prob = 1 / (1 + np.exp(-logit))
            
            # Generate binary outcomes
            y = np.random.binomial(1, prob)
            
            # Fit logistic regression
            X = x.reshape(-1, 1)
            model = LogisticRegression(penalty=None)
            model.fit(X, y)
            
            # Get predictions
            y_pred_prob = model.predict_proba(X)[:, 1]
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            
            # Create DataFrame for plotting
            data = pd.DataFrame({
                'x': x,
                'y': y,
                'probability': y_pred_prob
            })
            
            # Plot scatter and logistic curve
            fig = go.Figure()
            
            # Add scatter plot for outcomes
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name='Observed Outcomes',
                    marker=dict(
                        size=10,
                        color=y,
                        colorscale='Viridis',
                        line=dict(width=1)
                    )
                )
            )
            
            # Add logistic curve
            x_range = np.linspace(min(x), max(x), 100)
            X_range = x_range.reshape(-1, 1)
            y_range = model.predict_proba(X_range)[:, 1]
            
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=y_range,
                    mode='lines',
                    name='Logistic Curve',
                    line=dict(color='red', width=3)
                )
            )
            
            # Add individual probabilities if enabled
            if show_probabilities:
                for i in range(len(x)):
                    if y[i] == 1:
                        fig.add_shape(
                            type="line",
                            x0=x[i], y0=y[i],
                            x1=x[i], y1=y_pred_prob[i],
                            line=dict(color="green", width=1, dash="dash"),
                        )
                    else:
                        fig.add_shape(
                            type="line",
                            x0=x[i], y0=y[i],
                            x1=x[i], y1=y_pred_prob[i],
                            line=dict(color="red", width=1, dash="dash"),
                        )
            
            fig.update_layout(
                title=f"Logistic Regression (AUC = {roc_auc:.3f})",
                xaxis_title=x_label,
                yaxis_title=f"Probability of {y_label.split('(')[0].strip()}",
                yaxis=dict(range=[-0.05, 1.05]),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display model summary
            st.subheader("Model Summary")
            
            # Calculate odds ratio
            odds_ratio = np.exp(model.coef_[0][0])
            ci_width = 1.96 * np.sqrt(1/sample_size)  # Simplified CI calculation
            or_lower = odds_ratio * np.exp(-ci_width)
            or_upper = odds_ratio * np.exp(ci_width)
            
            st.write(f"Log-odds equation: log(p/(1-p)) = {model.intercept_[0]:.3f} + {model.coef_[0][0]:.3f} × {x_label}")
            st.write(f"Odds Ratio: {odds_ratio:.3f} (95% CI: {or_lower:.3f} - {or_upper:.3f})")
            st.write(f"AUC (Area Under ROC Curve): {roc_auc:.3f}")
            
            # Add ROC curve
            fig_roc = px.area(
                x=fpr, y=tpr,
                title=f'ROC Curve (AUC = {roc_auc:.3f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            
            fig_roc.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            
            fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_roc.update_xaxes(constrain='domain')
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Interpretation section
        st.subheader("Interpreting Logistic Regression Results")
        
        st.markdown(f"""
        ### Interpretation for {dataset_choice}:
        
        1. **The coefficient ({model.coef_[0][0]:.3f})** represents:
           - The change in log-odds of the outcome for each one-unit increase in {x_label}
           - This is not easily interpretable in its raw form
        
        2. **The odds ratio ({odds_ratio:.3f})** represents:
           - How much the odds of the outcome increase for each one-unit increase in {x_label}
           - An odds ratio > 1 means increasing {x_label} increases the likelihood of the outcome
           - An odds ratio < 1 would mean increasing {x_label} decreases the likelihood
           - For this model: Each one-unit increase in {x_label} multiplies the odds by {odds_ratio:.3f}
        
        3. **The intercept ({model.intercept_[0]:.3f})** represents:
           - The log-odds of the outcome when {x_label} is zero
           - This translates to a baseline probability of {1/(1+np.exp(-model.intercept_[0])):.3f}
        
        4. **AUC ({roc_auc:.3f})** represents:
           - The model's ability to discriminate between positive and negative outcomes
           - Ranges from 0.5 (no discrimination) to 1.0 (perfect discrimination)
           - This model has {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'fair' if roc_auc > 0.7 else 'poor'} discrimination
        """)
        
        st.subheader("Applications in Pharmacy")
        
        application_examples = {
            "Drug Response": """
            - Predicting patient response to medications based on drug concentration
            - Determining the probability of therapeutic success at different drug levels
            - Identifying the minimum effective concentration for a desired response probability
            - Personalizing dosing regimens based on individualized response probabilities
            """,
            "Adverse Event Risk": """
            - Quantifying how patient age affects the risk of medication-related adverse events
            - Developing risk prediction models for medication safety monitoring
            - Identifying high-risk patients who need additional monitoring or dose adjustments
            - Creating age-based dosing guidelines to minimize adverse event risk
            """,
            "Treatment Success": """
            - Modeling how medication adherence impacts treatment success
            - Identifying the critical adherence threshold associated with successful outcomes
            - Targeting adherence interventions based on probability of treatment success
            - Predicting individual patient outcomes based on their adherence patterns
            """
        }
        
        st.markdown(application_examples[dataset_choice])
    
    # Quiz section
    st.header("🧐 Test Your Understanding")
    
    if regression_type == "Linear Regression":
        
        quiz_q_options = [
            "-- Select an option --",
            "The y-value when x equals zero",
            "The change in y for a one-unit change in x",
            "The total variance explained by the model",
            "The typical error in predictions"
        ]
        quiz_q = st.radio(
            "What does the slope coefficient represent in linear regression?",
            quiz_q_options,
            index=0,
            key="quiz_q"
        )
        
        if quiz_q != quiz_q_options[0]:
            if quiz_q == quiz_q_options[2]:
                st.success("✅ Correct! The slope coefficient shows how much the dependent variable changes when the independent variable increases by one unit.")
            else:
                st.error("❌ Not correct. The slope coefficient (β₁) represents the change in the outcome variable for each one-unit increase in the predictor variable.")
    else:
        quiz_q2_options =[
            "-- Select an option --",
            "The probability of the outcome increases by 2.5",
            "The outcome is 2.5 times more likely to occur",
            "For each one-unit increase in the predictor, the odds of the outcome increase 2.5 times",
            "The model explains 2.5% of the variance"
            ]
        quiz_q2 = st.radio(
            "How do you interpret an odds ratio of 2.5 in logistic regression?",
            quiz_q2_options,
            index=0,    
            key="quiz_q"
        )   
        
        if quiz_q2 != quiz_q2_options[0]:
            if quiz_q2 == quiz_q2_options[3]:
                st.success("✅ Correct! The odds ratio tells us how much the odds of the outcome multiply for each one-unit increase in the predictor.")
            else:
                st.error("❌ Not correct. An odds ratio of 2.5 means that for each one-unit increase in the predictor variable, the odds of the outcome multiply by 2.5 times.")

    # Further reading
    st.header("Further Learning")
    st.markdown("""
    **Key Differences Between Linear and Logistic Regression:**
    
    | Feature | Linear Regression | Logistic Regression |
    |---------|-------------------|---------------------|
    | Outcome | Continuous | Binary (0/1) |
    | Equation | y = β₀ + β₁x + ε | log(p/(1-p)) = β₀ + β₁x |
    | Interpretation | Slope is change in outcome | Coefficient gives log-odds ratio |
    | Curve shape | Straight line | S-shaped (sigmoid) curve |
    | Key metrics | R², RMSE | Odds ratio, AUC |
    | Pharmacy examples | Drug concentration-effect, dose-response | Treatment success/failure, adverse event risk |
    
    **Further Reading:**
    - [Understanding Linear Regression for Pharmaceutical Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3127302/)
    - [Applications of Logistic Regression in Pharmacy Practice](https://www.pharmacytimes.com/)
    - [Interpreting Regression Results in Clinical Studies](https://jamanetwork.com/)
    """)

# Code Laboratory tab content
with code_tab:
    regression_code.app()