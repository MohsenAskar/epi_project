import streamlit as st

st.set_page_config(layout="wide", page_title="Important Definitions")
st.title("Important Definitions")
st.write("""
The definitions are sorted after the different modules

- **Data Distributions module**
    - ***Mean***: The average value in a set of numbers. For example, the mean of 2, 4, and 6 is (2+4+6)/3 = 4.
    
    - ***Standard Deviation (SD)***: A measure of how spread out numbers are from the average. A small SD means data points are close to the average; a large SD means they're more spread out. For instance, the blood pressure readings of patients on a medication might have a small or large SD depending on how consistently the drug works.
    
    - ***Sample Size (n)***: The number of individuals or observations included in a study. For example, if you measure the blood pressure of 50 patients, your sample size is 50.
    
    - ***Cumulative Distribution Function (CDF)***: Shows the probability that a value will be less than or equal to a certain number. For example, a CDF might show that 75% of patients have blood pressure readings below 140 mmHg.
    
    - ***Weibull Distribution***: A flexible statistical distribution often used to model time until an event occurs, like how long until a medication starts working or how long a patient survives with a certain condition.
    
    - ***Statistical power***: The ability of a study to detect an effect if one truly exists. Higher power means you're less likely to miss important effects. For example, a study with 1000 patients has more power to detect a rare side effect than a study with only 50 patients.
    
    - ***p-value (alpha (α))***: A number between 0 and 1 that helps determine if a result is statistically significant. Typically, if p < 0.05, we consider the result unlikely to have occurred by chance. For example, a p-value of 0.02 for a medication's effect would suggest the effect is probably real and not just random chance.
    
    - ***Exceedance Probability (EP)***: The probability that a value will exceed a certain threshold. For example, the chance that a patient's cholesterol level will be above 200 mg/dL after treatment.
    
    - ***Arithmetic Mean***: Another term for the average, where you add up all values and divide by the number of values.
    
    
- **Regression module**
    - ***Dependent variable (Y)***: The outcome you're trying to predict or explain. For example, in a study examining how drug dosage affects blood pressure, blood pressure is the dependent variable.
    
    - ***Independent variable (X)***: The factor you think affects or predicts the outcome. Using the same example, drug dosage would be the independent variable.
    
    - ***y-intercept***: The value of Y when X equals zero. For example, in a relationship between drug dosage and effect, it would be the baseline effect when no drug is given.
    
    - ***Coefficient***: A number that describes the relationship between variables. It tells you how much Y changes when X changes by one unit.
    
    - ***The slope coefficient***: Specifically tells you how much Y increases or decreases when X increases by one unit. For example, if the slope coefficient is 2, then blood pressure decreases by 2 mmHg for each additional mg of medication.
    
    - ***Error term (residual)***: The difference between the actual value and the predicted value. It represents what your model couldn't explain. For example, if you predicted a patient's blood pressure would be 130 mmHg, but it's actually 135 mmHg, the residual is 5 mmHg.
    
    - ***R-squared (R²)***: A measure from 0 to 1 that tells you how well your model explains the data. An R² of 0.7 means 70% of the variation in Y is explained by X. For example, an R² of 0.8 would mean that 80% of the differences in patient outcomes can be explained by the factors in your model.
    
    - ***Fit line (Regression line)***: The line that best describes the relationship between X and Y. It's the line that minimizes the errors between predicted and actual values.
    
    - ***RMSE (Root Mean Square Error)***: A measure of the typical size of the errors in your predictions. Smaller values mean better predictions. For example, an RMSE of 5 mmHg in blood pressure predictions means your typical prediction is off by about 5 mmHg.
    
    - ***Binary outcome***: A result that has only two possible values, like yes/no or success/failure. For example, whether a patient experiences a side effect (yes or no).
    
    - ***Sigmoid function***: An S-shaped curve used in logistic regression to model probabilities. It transforms any value to a number between 0 and 1, making it useful for predicting binary outcomes.
    
    - ***AUC (Area Under ROC Curve)***: A measure from 0 to 1 that tells you how well a model distinguishes between two outcomes (like disease/no disease). An AUC of 0.5 is no better than random guessing; an AUC of 1 is perfect prediction.
    
    - ***The odds ratio (OR)***: Compares the odds of an outcome in two groups. An OR of 1 means equal odds; OR > 1 means higher odds in the first group; OR < 1 means lower odds. For example, an OR of 2 might mean patients taking a drug have twice the odds of recovery compared to those not taking it.
    
- **Correlation module**
    - ***Continuous variables***: Variables that can take any numerical value within a range, like height, weight, blood pressure, or drug concentration in blood.
    
    - ***Categorical variables***: Variables that fall into distinct categories or groups, like gender (male/female), treatment group (drug A/drug B/placebo), or side effect status (present/absent).

- **Causal Inference module**
    - ***Propensity Score***: A single number that represents the probability of receiving a treatment based on a patient's characteristics. It helps compare similar patients who did and didn't receive a treatment. For example, comparing patients with a similar likelihood of receiving a certain drug, even if one group actually received it and the other didn't.
    
    - ***Instrumental Variables***: Variables that affect the treatment received but don't directly affect the outcome. They help establish causality when randomization isn't possible. For example, distance to a specialized hospital might affect whether a patient receives a certain treatment but doesn't directly affect their outcome.
    
- **Confounding module**
    - ***Crude Effect***: The observed relationship between a treatment and outcome without accounting for other factors that might influence this relationship. For example, seeing that patients taking a drug have better outcomes, without considering that they might also be younger or healthier to begin with.
    
    - ***Adjusted Effect***: The relationship between a treatment and outcome after accounting for other factors (confounders). For example, the effect of a drug after accounting for patients' age, sex, and other health conditions.
    
- **Stratification module**
    - ***Effect modification***: When the effect of a treatment varies across different groups. For example, a drug might work better in younger patients than in older ones.
    
    - ***Interaction***: When two or more factors combine to produce an effect that's different than what would be expected from each factor alone. For example, when a drug works especially well (or poorly) when combined with another medication.
    
- **Measures of Association module**
    - ***Risk-Based Measures***: Ways to quantify associations based on probabilities or risks, such as relative risk (comparing risk between groups) or risk difference (subtracting one risk from another).
    
    - ***Odds-Based Measures***: Ways to quantify associations based on odds, like the odds ratio, which compares the odds of an outcome between different groups.
    
    - ***Incidence***: The number of new cases of a condition that develop in a population over a specific time period. For example, the number of new diabetes cases diagnosed in a community during one year.
    
    - ***Incidence Proportion***: The proportion of people who develop a condition during a specific time period. For example, if 50 out of 1000 people develop a condition in one year, the incidence proportion is 50/1000 = 0.05 or 5%.
    
- **Effect Modification module**
    - ***Statistically significant***: When a result is unlikely to have occurred by chance. Usually indicated by a p-value less than 0.05 (5%). For example, if a drug shows improvement with p=0.03, we consider this statistically significant, meaning the improvement is probably real and not just due to random variation.
    
- **Epidemiological Study Designs module**
    - ***Cohort***: A group of individuals followed over time to observe outcomes. For example, following a group of patients taking a new medication for several years to monitor long-term effects.
    
- **Selection Bias module**
    - ***Selection Bias***: When the individuals studied don't properly represent the population of interest, leading to misleading results. For example, if a drug study only includes younger patients but the drug is meant for all ages, the results might not apply to older patients.
""")
