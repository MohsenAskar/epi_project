import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys
import os

st.set_page_config(layout="wide", page_title="Common Data Distributions in Epidemiology")

# Add path to import from interactive_code directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interactive_code'))

# Import the code lab module
import data_distributions_code

# Title Section
st.title("Common Data Distributions in Epidemiology")
st.markdown("""
This interactive module helps you understand the common probability distributions 
encountered in epidemiological research and public health data analysis.
""")

# Create tabs for Visualization and Code Lab
viz_tab, code_tab = st.tabs(["Interactive Visualization", "Code Laboratory"])

# Visualization Tab Content
with viz_tab:
    # Create main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Distribution Controls")
        
        # Distribution selection
        distribution_type = st.selectbox(
            "Select Distribution",
            [
                "Normal (Gaussian) Distribution",
                "Binomial Distribution",
                "Poisson Distribution",
                "Exponential Distribution",
                "Log-Normal Distribution",
                "Uniform Distribution"
            ]
        )
        
        # Parameters for each distribution
        if distribution_type == "Normal (Gaussian) Distribution":
            mean = st.slider("Mean (Average Value)", -10.0, 10.0, 0.0, 0.1)
            std_dev = st.slider("Standard Deviation (Spread of Values)", 0.1, 5.0, 1.0, 0.1)
            sample_size = st.slider("Sample Size (Number of Patients)", 100, 5000, 1000, 100)
            
            # Generate data
            data = np.random.normal(mean, std_dev, sample_size)
            
            # Display parameters
            st.write(f"**Parameters:**")
            st.write(f"- Mean (μ): {mean}")
            st.write(f"- Standard Deviation (σ): {std_dev}")
            
            # Calculate summary statistics
            actual_mean = np.mean(data)
            actual_median = np.median(data)
            actual_std = np.std(data)
            
            st.write(f"**Sample Statistics:**")
            st.write(f"- Sample Mean: {actual_mean:.2f}")
            st.write(f"- Sample Median (Middle Value): {actual_median:.2f}")
            st.write(f"- Sample Standard Deviation: {actual_std:.2f}")
            
        elif distribution_type == "Binomial Distribution":
            n_trials = st.slider("Number of Trials (n)", 1, 100, 20)
            p_success = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
            sample_size = st.slider("Sample Size", 100, 5000, 1000, 100)
            
            # Generate data
            data = np.random.binomial(n_trials, p_success, sample_size)
            
            # Display parameters
            st.write(f"**Parameters:**")
            st.write(f"- Number of Patients (n): {n_trials}")
            st.write(f"- Probability of Success (p): {p_success}")
            
            # Calculate summary statistics
            actual_mean = np.mean(data)
            actual_var = np.var(data)
            
            st.write(f"**Sample Statistics:**")
            st.write(f"- Sample Mean: {actual_mean:.2f} (Expected: {n_trials*p_success:.2f})")
            st.write(f"- Sample Variance: {actual_var:.2f} (Expected: {n_trials*p_success*(1-p_success):.2f})")
        
        elif distribution_type == "Poisson Distribution":
            rate = st.slider("Average Events per Time Period", 0.1, 20.0, 5.0, 0.1)
            sample_size = st.slider("Number of Time Periods", 100, 5000, 1000, 100)
            
            # Generate data
            data = np.random.poisson(rate, sample_size)
            
            # Display parameters
            st.write(f"**Parameters:**")
            st.write(f"- Rate (λ): {rate}")
            
            # Calculate summary statistics
            actual_mean = np.mean(data)
            actual_var = np.var(data)
            
            st.write(f"**Sample Statistics:**")
            st.write(f"- Sample Mean: {actual_mean:.2f} (Expected: {rate:.2f})")
            st.write(f"- Sample Variance: {actual_var:.2f} (Expected: {rate:.2f})")
        
        elif distribution_type == "Exponential Distribution":
            scale = st.slider("Scale (β = 1/λ)", 0.1, 10.0, 1.0, 0.1)
            sample_size = st.slider("Sample Size", 100, 5000, 1000, 100)
            
            # Generate data
            data = np.random.exponential(scale, sample_size)
            
            # Display parameters
            st.write(f"**Parameters:**")
            st.write(f"- Scale (β): {scale}")
            st.write(f"- Rate (λ): {1/scale:.2f}")
            
            # Calculate summary statistics
            actual_mean = np.mean(data)
            actual_var = np.var(data)
            
            st.write(f"**Sample Statistics:**")
            st.write(f"- Sample Mean: {actual_mean:.2f} (Expected: {scale:.2f})")
            st.write(f"- Sample Variance: {actual_var:.2f} (Expected: {scale**2:.2f})")
        
        elif distribution_type == "Log-Normal Distribution":
            mu = st.slider("Log-space Mean (μ)", -2.0, 2.0, 0.0, 0.1)
            sigma = st.slider("Log-space SD (Amount of Skew)", 0.1, 2.0, 0.5, 0.1)
            sample_size = st.slider("Sample Size", 100, 5000, 1000, 100)
            
            # Generate data
            data = np.random.lognormal(mu, sigma, sample_size)
            
            # Display parameters
            st.write(f"**Parameters:**")
            st.write(f"- Log-space Mean (μ): {mu}")
            st.write(f"- Log-space Standard Deviation (σ): {sigma}")
            
            # Calculate expected statistics in original space
            expected_mean = np.exp(mu + sigma**2/2)
            expected_median = np.exp(mu)
            expected_var = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
            
            st.write(f"**Theoretical Statistics:**")
            st.write(f"- Mean: {expected_mean:.2f}")
            st.write(f"- Median: {expected_median:.2f}")
            st.write(f"- Variance: {expected_var:.2f}")
            
            st.write(f"**Sample Statistics:**")
            st.write(f"- Sample Mean: {np.mean(data):.2f}")
            st.write(f"- Sample Median: {np.median(data):.2f}")
            st.write(f"- Sample Variance: {np.var(data):.2f}")
        
        elif distribution_type == "Uniform Distribution":
            lower = st.slider("Minimum Value (a)", -10.0, 10.0, 0.0, 0.1)
            upper = st.slider("Maximum Value (b)", lower + 0.1, 20.0, lower + 5.0, 0.1)
            sample_size = st.slider("Sample Size", 100, 5000, 1000, 100)
            
            # Generate data
            data = np.random.uniform(lower, upper, sample_size)
            
            # Display parameters
            st.write(f"**Parameters:**")
            st.write(f"- Minimum Value (a): {lower}")
            st.write(f"- Maximum Value (b): {upper}")
            
            # Calculate expected statistics
            expected_mean = (lower + upper) / 2
            expected_var = (upper - lower)**2 / 12
            
            st.write(f"**Theoretical Statistics:**")
            st.write(f"- Mean: {expected_mean:.2f}")
            st.write(f"- Variance: {expected_var:.2f}")
            
            st.write(f"**Sample Statistics:**")
            st.write(f"- Sample Mean: {np.mean(data):.2f}")
            st.write(f"- Sample Variance: {np.var(data):.2f}")

    with col2:
        st.header("Distribution Visualization")
        
        # Create the histogram
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=data,
            histnorm='probability density',
            name='Observed Data',
            marker_color='#3366CC',
            opacity=0.7,
            nbinsx=30
        ))
        
        # Add theoretical PDF for continuous distributions
        if distribution_type in ["Normal (Gaussian) Distribution", "Exponential Distribution", 
                                "Log-Normal Distribution", "Uniform Distribution"]:
            x_range = np.linspace(min(data), max(data), 1000)
            
            if distribution_type == "Normal (Gaussian) Distribution":
                y_range = stats.norm.pdf(x_range, mean, std_dev)
                pdf_name = f"Normal Expected Distribution (μ={mean}, σ={std_dev})"
            
            elif distribution_type == "Exponential Distribution":
                y_range = stats.expon.pdf(x_range, scale=scale)
                pdf_name = f"Exponential Expected Distribution (β={scale})"
            
            elif distribution_type == "Log-Normal Distribution":
                # Filter out potential extreme values for better visualization
                x_range = np.linspace(max(0.001, min(data)), 
                                    min(np.percentile(data, 99), max(data)), 1000)
                y_range = stats.lognorm.pdf(x_range, s=sigma, scale=np.exp(mu))
                pdf_name = f"Log-Normal Expected Distribution (μ={mu}, σ={sigma})"
            
            elif distribution_type == "Uniform Distribution":
                y_range = stats.uniform.pdf(x_range, loc=lower, scale=upper-lower)
                pdf_name = f"Uniform Expected Distribution (a={lower}, b={upper})"
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                name=pdf_name,
                line=dict(color='red', width=2)
            ))
        
        # Add PMF for discrete distributions
        elif distribution_type in ["Binomial Distribution", "Poisson Distribution"]:
            if distribution_type == "Binomial Distribution":
                x_values = np.arange(0, n_trials + 1)
                y_values = stats.binom.pmf(x_values, n_trials, p_success)
                pmf_name = f"Binomial PMF (n={n_trials}, p={p_success})"
            
            elif distribution_type == "Poisson Distribution":
                x_max = max(20, int(np.percentile(data, 99)))
                x_values = np.arange(0, x_max + 1)
                y_values = stats.poisson.pmf(x_values, rate)
                pmf_name = f"Poisson PMF (λ={rate})"
            
            fig.add_trace(go.Bar(
                x=x_values,
                y=y_values,
                name=pmf_name,
                marker_color='red',
                opacity=0.5
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{distribution_type} - Histogram with Theoretical Distribution",
            xaxis_title="Value",
            yaxis_title="Probability Density",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a second plot for CDF
        fig2 = go.Figure()
        
        # Sort data for empirical CDF
        sorted_data = np.sort(data)
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Add empirical CDF
        fig2.add_trace(go.Scatter(
            x=sorted_data,
            y=cumulative_prob,
            mode='lines',
            name='Empirical CDF',
            line=dict(color='#3366CC', width=2)
        ))
        
        # Add theoretical CDF
        if distribution_type == "Normal (Gaussian) Distribution":
            x_range = np.linspace(min(data), max(data), 1000)
            y_range = stats.norm.cdf(x_range, mean, std_dev)
            cdf_name = f"Normal CDF (μ={mean}, σ={std_dev})"
        
        elif distribution_type == "Binomial Distribution":
            x_range = np.arange(0, n_trials + 1)
            y_range = stats.binom.cdf(x_range, n_trials, p_success)
            cdf_name = f"Binomial CDF (n={n_trials}, p={p_success})"
        
        elif distribution_type == "Poisson Distribution":
            x_max = max(20, int(np.percentile(data, 99)))
            x_range = np.arange(0, x_max + 1)
            y_range = stats.poisson.cdf(x_range, rate)
            cdf_name = f"Poisson CDF (λ={rate})"
        
        elif distribution_type == "Exponential Distribution":
            x_range = np.linspace(min(data), max(data), 1000)
            y_range = stats.expon.cdf(x_range, scale=scale)
            cdf_name = f"Exponential CDF (β={scale})"
        
        elif distribution_type == "Log-Normal Distribution":
            x_range = np.linspace(max(0.001, min(data)), 
                                min(np.percentile(data, 99), max(data)), 1000)
            y_range = stats.lognorm.cdf(x_range, s=sigma, scale=np.exp(mu))
            cdf_name = f"Log-Normal CDF (μ={mu}, σ={sigma})"
        
        elif distribution_type == "Uniform Distribution":
            x_range = np.linspace(min(data), max(data), 1000)
            y_range = stats.uniform.cdf(x_range, loc=lower, scale=upper-lower)
            cdf_name = f"Uniform CDF (a={lower}, b={upper})"
        
        fig2.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name=cdf_name,
            line=dict(color='red', width=2)
        ))
        
        # Update layout
        fig2.update_layout(
            title=f"{distribution_type} - Cumulative Distribution Function",
            xaxis_title="Value",
            yaxis_title="Cumulative Probability",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # Educational content
    st.header("About Data Distributions in Epidemiology")

    # Create expanders for each distribution
    with st.expander("Normal (Gaussian) Distribution"):
        st.markdown("""
        ### Normal Distribution
        
        The Normal (or Gaussian) distribution is a continuous probability distribution that is symmetric around its mean, with data more likely to be close to the mean.
        
        **Parameters:**
        - μ (mean): Center of the distribution
        - σ (standard deviation): Spread of the distribution
        
        **Properties:**
        - Symmetric bell-shaped curve
        - 68% of data within 1σ of the mean
        - 95% of data within 2σ of the mean
        - 99.7% of data within 3σ of the mean (the "empirical rule")
        
        **Applications in Epidemiology:**
        - Heights and weights in a population
        - Blood pressure measurements
        - Laboratory measurement errors
        - Statistical tests that assume normality (t-tests, ANOVA)
        
        **Central Limit Theorem:**
        Even when the underlying data is not normally distributed, the sampling distribution of the mean approaches a normal distribution as sample size increases.
        """)

    with st.expander("Binomial Distribution"):
        st.markdown("""
        ### Binomial Distribution
        
        The Binomial distribution models the number of successes in a fixed number of independent trials, each with the same probability of success.
        
        **Parameters:**
        - n (number of trials): The number of independent experiments
        - p (probability of success): The probability of success in a single trial
        
        **Properties:**
        - Discrete distribution (whole numbers only)
        - Mean = n×p
        - Variance = n×p×(1-p)
        
        **Applications in Epidemiology:**
        - Number of people infected in a sample of fixed size
        - Count of individuals with a specific disease in a group
        - Vaccine effectiveness studies (success/failure outcomes)
        - Number of patients responding to treatment
        
        **Note:**
        For rare events (small p) and large number of trials (large n), the Binomial distribution can be approximated by the Poisson distribution with λ = n×p.
        """)

    with st.expander("Poisson Distribution"):
        st.markdown("""
        ### Poisson Distribution
        
        The Poisson distribution models the number of events occurring in a fixed interval of time or space, when these events happen independently at a constant average rate.
        
        **Parameter:**
        - λ (lambda): The average number of events in the interval
        
        **Properties:**
        - Discrete distribution for count data
        - Mean = λ
        - Variance = λ
        - As λ increases, the Poisson distribution approaches a Normal distribution
        
        **Applications in Epidemiology:**
        - Disease incidence rates (cases per time period)
        - Number of deaths in a hospital per day
        - Number of cancer cases in a geographic region
        - Rare disease outbreaks
        - Count of mutations or genetic events
        
        **When to Use:**
        The Poisson distribution is particularly useful when:
        1. Events occur independently
        2. Events occur at a constant average rate
        3. Two events cannot occur at exactly the same time
        """)

    with st.expander("Exponential Distribution"):
        st.markdown("""
        ### Exponential Distribution
        
        The Exponential distribution models the time between events in a Poisson process, where events occur continuously and independently at a constant average rate.
        
        **Parameter:**
        - β (scale parameter): Average time between events (1/λ)
        - λ (rate parameter): Average number of events per unit time
        
        **Properties:**
        - Continuous distribution
        - "Memoryless" property: P(T > s+t | T > s) = P(T > t)
        - Mean = β = 1/λ
        - Variance = β² = 1/λ²
        
        **Applications in Epidemiology:**
        - Survival analysis (time until death or disease progression)
        - Length of hospital stays
        - Time between infection events
        - Waiting time between disease occurrences
        - Time until equipment failure in medical devices
        
        **Relationship to Poisson:**
        If events occur according to a Poisson process with rate λ, then the time between consecutive events follows an Exponential distribution with parameter λ.
        """)

    with st.expander("Log-Normal Distribution (Skewed Data)"):
        st.markdown("""
        ### Log-Normal Distribution
        
        A continuous probability distribution where the logarithm of the random variable is normally distributed. It results from the multiplicative product of many independent positive random variables.
        
        **Parameters:**
        - μ (log-scale mean): Mean of the variable's natural logarithm
        - σ (log-scale standard deviation): Standard deviation of the variable's natural logarithm
        
        **Properties:**
        - Always positive values
        - Right-skewed distribution
        - Median = e^μ
        - Mean = e^(μ+σ²/2)
        - Variance = (e^σ² - 1)e^(2μ+σ²)
        
        **Applications in Epidemiology:**
        - Biological measurements (e.g., antibody levels, pathogen concentrations)
        - Incubation periods of infectious diseases
        - Environmental exposure data
        - Healthcare costs and length of hospital stays
        - Drug concentration in the body over time
        
        **Why it Occurs:**
        Many biological processes involve multiplicative effects or growth processes, which naturally lead to log-normal distributions.
        """)

    with st.expander("Uniform Distribution"):
        st.markdown("""
        ### Uniform Distribution
        
        The Uniform distribution represents equal probability across an interval. Every value within the range has the same likelihood of occurring.
        
        **Parameters:**
        - a: Lower bound
        - b: Upper bound
        
        **Properties:**
        - Constant probability density function
        - Mean = (a+b)/2
        - Variance = (b-a)²/12
        
        **Applications in Epidemiology:**
        - Random sampling procedures
        - Modeling complete uncertainty about exposure levels
        - Prior distributions in Bayesian analyses when no prior knowledge exists
        - Generating random numbers for simulations
        - Initial value ranges in optimization algorithms
        
        **Usage:**
        While less common for modeling actual epidemiological data, the uniform distribution is fundamental for:
        1. Simulation studies
        2. Statistical tests (e.g., randomization tests)
        3. Methodological research
        """)

    # Interactive Quiz
    st.header("Test Your Understanding")

    question = st.radio(
        "Which of the following data would most likely follow a Poisson distribution?",
        [
            "Heights of individuals in a population",
            "Number of new COVID-19 cases per day in a small town",
            "Time until recovery after treatment for all patients",
            "Systolic blood pressure measurements in adults"
        ]
    )

    if question == "Number of new COVID-19 cases per day in a small town":
        st.success("Correct! The Poisson distribution is well-suited for modeling count data of independent events occurring in a fixed time period, like new cases per day.")
    else:
        st.error("Not quite right. Think about which answer involves counting discrete events occurring in a fixed time period.")

    # Example application
    st.header("Real-world Application Example")

    application_select = st.selectbox(
        "Select an epidemiological application:",
        [
            "Disease Outbreak Simulation",
            "Survival Analysis",
            "Clinical Trial Success Probability",
            "Exposure Risk Assessment"
        ]
    )

    if application_select == "Disease Outbreak Simulation":
        st.markdown("""
        ### Disease Outbreak Simulation
        
        In this example, we'll simulate the daily case counts during a disease outbreak using a Poisson process,
        but with a time-varying rate parameter to represent the epidemic curve.
        """)
        
        # Parameters
        days = 90
        peak_day = st.slider("Peak Day of Outbreak", 10, 70, 30)
        peak_cases = st.slider("Peak Daily Cases", 10, 200, 50)
        r0 = st.slider("Basic Reproduction Number (R₀)", 0.5, 5.0, 2.5, 0.1)
        
        # Generate epidemic curve (log-normal shape is common for outbreaks)
        x = np.arange(1, days + 1)
        epidemic_curve = peak_cases * np.exp(-(np.log(x) - np.log(peak_day))**2 / (2 * (r0 / 3)**2))
        
        # Generate daily cases using Poisson distribution
        np.random.seed(42)  # For reproducibility
        daily_cases = np.random.poisson(epidemic_curve)
        
        # Create dataframe
        outbreak_df = pd.DataFrame({
            'Day': x,
            'Expected Cases': epidemic_curve,
            'Actual Cases': daily_cases,
            'Cumulative Cases': np.cumsum(daily_cases)
        })
        
        # Plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=outbreak_df['Day'],
            y=outbreak_df['Expected Cases'],
            mode='lines',
            name='Expected Cases (Poisson λ)',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Bar(
            x=outbreak_df['Day'],
            y=outbreak_df['Actual Cases'],
            name='Daily Cases (Poisson random variable)',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Simulated Disease Outbreak (Daily Cases)",
            xaxis_title="Day of Outbreak",
            yaxis_title="Number of Cases",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative curve
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=outbreak_df['Day'],
            y=outbreak_df['Cumulative Cases'],
            mode='lines',
            name='Cumulative Cases',
            line=dict(color='darkblue', width=2)
        ))
        
        fig2.update_layout(
            title="Simulated Disease Outbreak (Cumulative Cases)",
            xaxis_title="Day of Outbreak",
            yaxis_title="Cumulative Number of Cases",
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **Discussion**:
        
        This simulation demonstrates how the Poisson distribution can model daily case counts in an outbreak.
        The expected number of cases (λ parameter) follows an epidemic curve, but the actual observed cases
        show random variation around this expectation.
        
        Key points:
        1. The Poisson distribution captures the random nature of disease occurrence
        2. As λ increases (at the peak), the variance increases as well
        3. The cumulative case curve resembles an S-shape, typical of epidemic growth and control
        
        In real outbreak analysis, this type of modeling helps with:
        - Forecasting future cases
        - Estimating the reproductive number
        - Planning healthcare resource allocation
        """)

    elif application_select == "Survival Analysis":
        st.markdown("""
        ### Survival Analysis in Epidemiology
        
        This example demonstrates how the Exponential and Weibull distributions can model survival times in a clinical setting.
        """)
        
        # Parameters
        n_patients = 200
        followup_years = 5
        
        # Treatment options
        treatment = st.radio(
            "Select treatment group to simulate:",
            ["Standard Treatment", "New Treatment"]
        )
        
        if treatment == "Standard Treatment":
            # Parameters for exponential distribution
            rate_param = 0.4  # hazard rate
            scale_param = 1/rate_param  # scale parameter for exponential
            
            # Generate survival times
            np.random.seed(42)
            survival_times = np.random.exponential(scale=scale_param, size=n_patients)
            dist_name = "Exponential"
            median_survival = np.log(2) * scale_param
        else:
            # Parameters for Weibull distribution (better survival)
            shape_param = 0.8  # shape parameter < 1 means decreasing hazard over time
            scale_param = 3.0  # scale parameter
            
            # Generate survival times from Weibull
            np.random.seed(42)
            survival_times = scale_param * np.random.weibull(shape_param, n_patients)
            dist_name = "Weibull"
            median_survival = scale_param * (np.log(2))**(1/shape_param)
        
        # Censoring - some patients are still alive at end of study
        censored = survival_times > followup_years
        observed_times = np.minimum(survival_times, followup_years)
        
        # Calculate Kaplan-Meier estimate
        sorted_times = np.sort(observed_times)
        events = np.array([not c for c in censored[np.argsort(observed_times)]])
        n_risk = np.arange(n_patients, 0, -1)
        
        # Calculate survival function
        km_survival = np.ones(n_patients)
        for i in range(n_patients):
            if events[i]:
                km_survival[i:] *= (1 - 1/n_risk[i])
        
        # Plot
        fig = go.Figure()
        
        # Add KM curve
        fig.add_trace(go.Scatter(
            x=sorted_times,
            y=km_survival,
            mode='lines',
            name='Kaplan-Meier Estimate',
            line=dict(color='blue', width=2)
        ))
        
        # Add censored points
        censored_times = sorted_times[~events]
        censored_survival = km_survival[~events]
        
        fig.add_trace(go.Scatter(
            x=censored_times,
            y=censored_survival,
            mode='markers',
            name='Censored',
            marker=dict(color='red', symbol='x', size=8)
        ))
        
        # Add theoretical curve
        t_range = np.linspace(0, followup_years, 100)
        if dist_name == "Exponential":
            s_range = np.exp(-t_range/scale_param)
        else:
            s_range = np.exp(-(t_range/scale_param)**shape_param)
        
        fig.add_trace(go.Scatter(
            x=t_range,
            y=s_range,
            mode='lines',
            name=f'Theoretical {dist_name}',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Add median survival
        if median_survival <= followup_years:
            fig.add_shape(
                type="line",
                x0=0, y0=0.5, x1=median_survival, y1=0.5,
                line=dict(color="gray", width=1, dash="dot")
            )
            
            fig.add_shape(
                type="line",
                x0=median_survival, y0=0, x1=median_survival, y1=0.5,
                line=dict(color="gray", width=1, dash="dot")
            )
            
            fig.add_annotation(
                x=median_survival,
                y=0.05,
                text=f"Median Survival: {median_survival:.2f} years",
                showarrow=False,
                yshift=10
            )
        
        fig.update_layout(
            title=f"Survival Curve for {treatment} (n={n_patients})",
            xaxis_title="Time (Years)",
            yaxis_title="Survival Probability",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add histogram of survival times
        fig2 = go.Figure()
        
        fig2.add_trace(go.Histogram(
            x=observed_times,
            histnorm='probability density',
            name='Observed Times',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add theoretical PDF
        if dist_name == "Exponential":
            pdf = stats.expon.pdf(t_range, scale=scale_param)
            label = f"Exponential Expected Distribution (rate={rate_param:.2f})"
        else:
            # Weibull PDF
            pdf = (shape_param/scale_param) * (t_range/scale_param)**(shape_param-1) * np.exp(-(t_range/scale_param)**shape_param)
            label = f"Weibull Expected Distribution (shape={shape_param:.2f}, scale={scale_param:.2f})"
        
        fig2.add_trace(go.Scatter(
            x=t_range,
            y=pdf,
            mode='lines',
            name=label,
            line=dict(color='red', width=2)
        ))
        
        fig2.update_layout(
            title=f"Distribution of Survival Times ({treatment})",
            xaxis_title="Time (Years)",
            yaxis_title="Probability Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown(f"""
        **Survival Analysis Results:**
        
        - Treatment: {treatment}
        - Distribution: {dist_name}
        - Number of patients: {n_patients}
        - Median survival time: {median_survival:.2f} years
        - Censored observations: {sum(censored)} ({sum(censored)/n_patients*100:.1f}%)
        
        **Discussion:**
        
        This example demonstrates how survival data in epidemiology often follows specific distributions:
        
        1. The **Exponential distribution** models survival with a constant hazard rate (the Standard Treatment in this simulation)
        2. The **Weibull distribution** can model increasing or decreasing hazard over time (the New Treatment)
        
        In real-world epidemiology, survival analysis helps:
        - Compare treatments in clinical trials
        - Estimate prognosis for patients
        - Identify risk factors for mortality or disease progression
        - Account for censored data (patients lost to follow-up or still alive at study end)
        """)

    elif application_select == "Clinical Trial Success Probability":
        st.markdown("""
        ### Clinical Trial Success Probability
        
        This example demonstrates using the Binomial distribution to calculate the probability of success in a clinical trial.
        """)
        
        # Parameters
        sample_size = st.slider("Sample Size (n)", 10, 1000, 100)
        true_effect = st.slider("True Treatment Effect (%)", 0, 50, 15)
        control_rate = st.slider("Control Group Success Rate (%)", 10, 90, 40)
        treatment_rate = control_rate + true_effect
        
        # Calculate trial success probability
        control_p = control_rate / 100
        treatment_p = treatment_rate / 100
        
        # Simulate many trials
        n_simulations = 10000
        np.random.seed(42)
        
        control_outcomes = np.random.binomial(sample_size, control_p, n_simulations)
        treatment_outcomes = np.random.binomial(sample_size, treatment_p, n_simulations)
        
        # Calculate p-values using chi-square test
        chi_square_stats = np.zeros(n_simulations)
        p_values = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            # Create contingency table
            table = np.array([
                [control_outcomes[i], sample_size - control_outcomes[i]],
                [treatment_outcomes[i], sample_size - treatment_outcomes[i]]
            ])
            
            # Expected values under null hypothesis
            row_sums = np.sum(table, axis=1)
            col_sums = np.sum(table, axis=0)
            total = np.sum(table)
            expected = np.outer(row_sums, col_sums) / total
            
            # Chi-square statistic
            chi_square = np.sum((table - expected)**2 / expected)
            chi_square_stats[i] = chi_square
            
            # p-value
            p_values[i] = 1 - stats.chi2.cdf(chi_square, df=1)
        
        # Calculate power (probability of rejecting null when alternative is true)
        alpha = 0.05
        power = np.mean(p_values < alpha)
        
        # Display results
        st.markdown(f"""
        **Trial Parameters:**
        - Control group success rate: {control_rate}%
        - Treatment group success rate: {treatment_rate}%
        - True treatment effect: {true_effect}%
        - Sample size per group: {sample_size}
        
        **Results:**
        - Statistical power: {power*100:.1f}%
        - Probability of trial success: {power*100:.1f}%
        """)
        
        # Plot histogram of p-values
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=p_values,
            histnorm='probability',
            cumulative_enabled=False,
            marker_color='lightblue',
            opacity=0.7,
            name='p-values'
        ))
        
        # Add vertical line at alpha
        fig.add_shape(
            type="line",
            x0=alpha, y0=0, x1=alpha, y1=0.4,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=alpha,
            y=0.3,
            text=f"α = {alpha}",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30
        )
        
        fig.update_layout(
            title="Distribution of p-values Across Simulated Trials",
            xaxis_title="p-value",
            yaxis_title="Probability",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot sample size vs power curve
        sample_sizes = np.arange(10, 501, 10)
        powers = []
        
        for n in sample_sizes:
            # Calculate power using normal approximation
            p_pooled = (control_p * n + treatment_p * n) / (2 * n)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n + 1/n))
            effect = treatment_p - control_p
            z = effect / se
            power_approx = 1 - stats.norm.cdf(1.96 - z)
            powers.append(power_approx)
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=sample_sizes,
            y=powers,
            mode='lines',
            name='Power',
            line=dict(color='blue', width=2)
        ))
        
        # Highlight current sample size
        fig2.add_trace(go.Scatter(
            x=[sample_size],
            y=[power],
            mode='markers',
            name='Current Design',
            marker=dict(color='red', size=10)
        ))
        
        # Add horizontal line at 80% power
        fig2.add_shape(
            type="line",
            x0=min(sample_sizes), y0=0.8, x1=max(sample_sizes), y1=0.8,
            line=dict(color="green", width=1, dash="dash")
        )
        
        fig2.add_annotation(
            x=max(sample_sizes) * 0.9,
            y=0.81,
            text="80% Power",
            showarrow=False
        )
        
        fig2.update_layout(
            title="Sample Size vs. Statistical Power",
            xaxis_title="Sample Size per Group",
            yaxis_title="Power (Probability of Detecting True Effect)",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **Discussion:**
        
        This simulation demonstrates how the Binomial distribution is used to model success/failure outcomes in clinical trials.
        
        Key points:
        1. The power of a study increases with sample size and effect size
        2. Statistical power represents the probability of detecting a true effect when it exists
        3. In trial design, epidemiologists typically aim for at least 80% power
        
        In real clinical trial planning:
        - Sample size calculations use the Binomial distribution properties
        - Power analysis helps determine the required number of participants
        - The observed treatment effect follows a Binomial distribution
        """)

    elif application_select == "Exposure Risk Assessment":
        st.markdown("""
        ### Environmental Exposure Risk Assessment
        
        This example demonstrates how the Log-Normal distribution is used to model environmental exposure data
        and calculate risk of exceeding health-based thresholds.
        """)
        
        # Parameters
        log_mean = st.slider("Log-scale Mean (μ)", -1.0, 3.0, 1.0, 0.1)
        log_sd = st.slider("Log-scale Standard Deviation (σ)", 0.1, 2.0, 0.8, 0.1)
        threshold = st.slider("Health-based Threshold Level", 1, 50, 15)
        sample_size = 1000
        
        # Generate exposure data
        np.random.seed(42)
        exposure_data = np.random.lognormal(mean=log_mean, sigma=log_sd, size=sample_size)
        
        # Calculate statistics
        mean_exposure = np.exp(log_mean + (log_sd**2)/2)
        median_exposure = np.exp(log_mean)
        mode_exposure = np.exp(log_mean - log_sd**2)
        
        # Calculate exceedance probability
        exceedance_prob = 1 - stats.lognorm.cdf(threshold, s=log_sd, scale=np.exp(log_mean))
        exceedance_count = np.sum(exposure_data > threshold)
        
        # Plot histogram with threshold
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=exposure_data,
            histnorm='probability density',
            name='Exposure Data',
            marker_color='lightblue',
            opacity=0.7,
            nbinsx=30
        ))
        
        # Add theoretical PDF
        x_range = np.linspace(max(0.001, min(exposure_data)), 
                            min(np.percentile(exposure_data, 99.5), max(exposure_data) * 1.5), 
                            1000)
        pdf_values = stats.lognorm.pdf(x_range, s=log_sd, scale=np.exp(log_mean))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=pdf_values,
            mode='lines',
            name=f'Log-Normal Expected Distribution (μ={log_mean:.1f}, σ={log_sd:.1f})',
            line=dict(color='red', width=2)
        ))
        
        # Add vertical line at threshold
        fig.add_shape(
            type="line",
            x0=threshold, y0=0, x1=threshold, y1=max(pdf_values) * 1.1,
            line=dict(color="green", width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=threshold * 1.1,
            y=max(pdf_values) * 0.8,
            text=f"Threshold = {threshold}",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30
        )
        
        # Shade the area above threshold
        exceed_x = x_range[x_range >= threshold]
        exceed_y = stats.lognorm.pdf(exceed_x, s=log_sd, scale=np.exp(log_mean))
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([[threshold], exceed_x, [exceed_x[-1]], [threshold]]),
            y=np.concatenate([[0], exceed_y, [0], [0]]),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255, 0, 0, 0)"),
            name=f"Exceedance Probability: {exceedance_prob:.1%}"
        ))
        
        fig.update_layout(
            title="Environmental Exposure Distribution",
            xaxis_title="Exposure Level",
            yaxis_title="Probability Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        st.markdown(f"""
        **Exposure Statistics:**
        
        - Arithmetic Mean: {mean_exposure:.2f}
        - Median: {median_exposure:.2f}
        - Mode: {mode_exposure:.2f}
        - Threshold: {threshold}
        - Probability of exceeding threshold: {exceedance_prob:.1%}
        - Observed exceedances: {exceedance_count} out of {sample_size} ({exceedance_count/sample_size:.1%})
        
        **Risk Interpretation:**
        
        Based on this exposure model, approximately {exceedance_prob:.1%} of the population would be 
        exposed to levels exceeding the health-based threshold of {threshold}.
        """)
        
        # Add log-probability plot
        fig2 = go.Figure()
        
        # Sort data for empirical CDF
        sorted_data = np.sort(exposure_data)
        log_sorted = np.log(sorted_data)
        
        # Use probit scale (inverse normal CDF) for y-axis
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / (len(sorted_data) + 1)  # add 1 to avoid 0 and 1
        probit_values = stats.norm.ppf(cumulative_prob)
        
        fig2.add_trace(go.Scatter(
            x=log_sorted,
            y=probit_values,
            mode='markers',
            name='Log-Exposure Data',
            marker=dict(color='blue', size=4)
        ))
        
        # Add theoretical line
        theory_x = np.linspace(min(log_sorted), max(log_sorted), 100)
        theory_y = (theory_x - log_mean) / log_sd
        
        fig2.add_trace(go.Scatter(
            x=theory_x,
            y=theory_y,
            mode='lines',
            name='Theoretical Line',
            line=dict(color='red', width=2)
        ))
        
        fig2.update_layout(
            title="Log-Probability Plot (Test for Log-Normality)",
            xaxis_title="Log(Exposure Level)",
            yaxis_title="Probit Scale (Standard Normal Quantiles)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **Discussion:**
        
        This example demonstrates why the Log-Normal distribution is commonly used in environmental epidemiology:
        
        1. Environmental exposures are often right-skewed and cannot be negative
        2. The Log-Normal distribution works well for multiplicative processes common in exposure pathways
        3. It allows for calculating the probability of exceeding health-based thresholds
        
        The log-probability plot is a diagnostic tool - if the data follows a Log-Normal distribution,
        the points should approximately fall on a straight line.
        
        In real exposure assessment, epidemiologists use these methods to:
        - Characterize population exposure distributions
        - Identify high-risk groups
        - Set regulatory standards
        - Design appropriate interventions
        """)

# Code Lab Tab Content
with code_tab:
    # We're using the imported function from the data_distributions_code module
    data_distributions_code.app()
    
# Further resources
st.markdown("""---""")
st.header("Further Resources")
st.markdown("""
- [CDC Principles of Epidemiology](https://www.cdc.gov/csels/dsepd/ss1978/index.html)
- [WHO Epidemiology Training Resources](https://www.who.int/data/gho/info/training-courses)
- [Introduction to Statistical Distributions by UCLA](https://stats.oarc.ucla.edu/other/mult-pkg/distribution/)
- [R Epidemiology Package (epitools)](https://cran.r-project.org/web/packages/epitools/index.html)
- [Johns Hopkins University Epidemiology Courses](https://www.jhsph.edu/courses/?dept=Epidemiology)
""")



