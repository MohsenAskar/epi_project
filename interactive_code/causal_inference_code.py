import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from contextlib import redirect_stdout
import sys
import traceback

def app():
    st.title("Interactive Coding Laboratory: Causal Inference and DAGs")
    
    st.markdown("""
    ## Learn by Coding: Causal Inference and Directed Acyclic Graphs (DAGs)
    
    This interactive laboratory allows you to explore concepts of causal inference 
    through directed acyclic graphs (DAGs) and simple simulations. Modify the example 
    code to see how different causal structures affect relationships between variables.
    
    Choose a topic to explore:
    """)
    
    # Simplified topic selection for first-year students
    topic = st.selectbox(
        "Select a topic:",
        ["Basic DAG Concepts", 
         "Causal Effects Estimation"]
    )
    
    # Display the selected topic
    if topic == "Basic DAG Concepts":
        basic_dag_concepts_lesson()
    elif topic == "Causal Effects Estimation":
        causal_effects_estimation_lesson()

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
                'nx': nx,
                'plt': plt,
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

def basic_dag_concepts_lesson():
    st.header("Basic DAG Concepts")
    
    st.markdown("""
    ### Understanding Directed Acyclic Graphs (DAGs)
    
    DAGs are visual tools for representing causal relationships between variables. 
    They consist of nodes (variables) connected by directed arrows indicating causation.
    
    Let's explore different causal structures and their implications:
    """)
    
    # DAG scenario selector
    dag_scenario = st.selectbox(
        "Choose a causal structure to explore:",
        ["Confounding", "Mediation", "Collider", "M-Bias"]
    )
    
    # Display different code examples based on the selected DAG scenario
    if dag_scenario == "Confounding":
        initial_code = """# Creating and simulating a Confounding DAG
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
confounding_strength = 0.5   # How strongly the confounder affects exposure and outcome
causal_effect = 0.3          # True causal effect of exposure on outcome
sample_size = 1000           # Number of observations

# Set random seed for reproducibility
np.random.seed(42)

# Create DAG using NetworkX
G = nx.DiGraph()

# Add nodes
G.add_node("Confounder", pos=(0, 1))
G.add_node("Exposure", pos=(-1, 0))
G.add_node("Outcome", pos=(1, 0))

# Add edges (arrows showing causation)
G.add_edges_from([
    ('Confounder', 'Exposure'),  # Confounder affects exposure
    ('Confounder', 'Outcome'),   # Confounder affects outcome
    ('Exposure', 'Outcome')      # Exposure affects outcome
])

# Draw the DAG
plt.figure(figsize=(8, 6))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
        font_size=12, font_weight='bold', arrows=True, 
        arrowsize=20, edge_color='gray')

# Add to output_vars to display it
output_vars['fig_dag'] = plt.gcf()

# Now let's simulate data following this causal structure
# 1. Generate the confounder
confounder = np.random.normal(0, 1, sample_size)

# 2. Generate exposure based on confounder
exposure_prob = 1 / (1 + np.exp(-(confounding_strength * confounder)))
exposure = np.random.binomial(1, exposure_prob, sample_size)

# 3. Generate outcome based on both confounder and exposure
outcome = (causal_effect * exposure) + (confounding_strength * confounder) + np.random.normal(0, 1, sample_size)

# Create a dataframe
data = pd.DataFrame({
    'Confounder': confounder,
    'Exposure': exposure,
    'Outcome': outcome
})

# Calculate associations
# Unadjusted association (not controlling for confounder)
unadjusted = np.mean(outcome[exposure == 1]) - np.mean(outcome[exposure == 0])

# Adjusted association (controlling for confounder)
# Using a simple linear regression model
import statsmodels.api as sm
X = sm.add_constant(data[['Exposure', 'Confounder']])
model = sm.OLS(data['Outcome'], X).fit()
adjusted = model.params['Exposure']

# Print results
print("Confounding Scenario Results:")
print("-" * 40)
print(f"True causal effect: {causal_effect:.3f}")
print(f"Unadjusted association: {unadjusted:.3f}")
print(f"Adjusted association (controlling for confounder): {adjusted:.3f}")

if abs(unadjusted - causal_effect) > abs(adjusted - causal_effect):
    print("\\nThe adjusted estimate is closer to the true causal effect.")
    print("This demonstrates how controlling for a confounder reduces bias.")
else:
    print("\\nUnexpected result! Check if the simulation parameters are reasonable.")

# Create visualization of the relationships
# 1. Box plot of outcome by exposure status
fig1 = plt.figure(figsize=(8, 6))
plt.boxplot([outcome[exposure == 0], outcome[exposure == 1]], 
            labels=['Unexposed', 'Exposed'])
plt.title('Unadjusted: Outcome by Exposure Status')
plt.ylabel('Outcome')
output_vars['fig1'] = fig1

# 2. Scatter plot showing confounder-outcome relationship
fig2 = plt.figure(figsize=(8, 6))
plt.scatter(confounder, outcome, c=exposure, cmap='coolwarm', alpha=0.6)
plt.title('Relationship Between Confounder and Outcome')
plt.xlabel('Confounder')
plt.ylabel('Outcome')
plt.colorbar(label='Exposure')
output_vars['fig2'] = fig2

# 3. Stratified analysis - show exposure-outcome relationship in different confounder strata
# Create confounder strata (low, medium, high)
data['Confounder_Strata'] = pd.qcut(data['Confounder'], 3, labels=['Low', 'Medium', 'High'])

# Calculate the exposure-outcome association in each stratum
stratum_effects = []
for stratum in ['Low', 'Medium', 'High']:
    stratum_data = data[data['Confounder_Strata'] == stratum]
    stratum_effect = np.mean(stratum_data[stratum_data['Exposure'] == 1]['Outcome']) - np.mean(stratum_data[stratum_data['Exposure'] == 0]['Outcome'])
    stratum_effects.append({'Stratum': stratum, 'Effect': stratum_effect})

stratum_df = pd.DataFrame(stratum_effects)
print("\\nStratified Analysis (Exposure Effect by Confounder Level):")
for _, row in stratum_df.iterrows():
    print(f"{row['Stratum']} confounder: {row['Effect']:.3f}")

# Bar chart of exposure effect by confounder stratum
fig3 = plt.figure(figsize=(8, 6))
plt.bar(stratum_df['Stratum'], stratum_df['Effect'])
plt.axhline(y=causal_effect, color='r', linestyle='--', label='True Causal Effect')
plt.title('Exposure Effect by Confounder Stratum')
plt.ylabel('Exposure Effect on Outcome')
plt.legend()
output_vars['fig3'] = fig3
"""
    elif dag_scenario == "Mediation":
        initial_code = """# Creating and simulating a Mediation DAG
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
direct_effect = 0.3           # Direct effect of exposure on outcome
indirect_effect = 0.2         # Indirect effect through mediator
mediation_strength = 0.6      # How strongly exposure affects mediator
sample_size = 1000            # Number of observations

# Set random seed for reproducibility
np.random.seed(42)

# Create DAG using NetworkX
G = nx.DiGraph()

# Add nodes
G.add_node("Exposure", pos=(-1, 0))
G.add_node("Mediator", pos=(0, 0))
G.add_node("Outcome", pos=(1, 0))

# Add edges (arrows showing causation)
G.add_edges_from([
    ('Exposure', 'Mediator'),   # Exposure affects mediator
    ('Mediator', 'Outcome'),    # Mediator affects outcome
    ('Exposure', 'Outcome')     # Direct effect of exposure on outcome
])

# Draw the DAG
plt.figure(figsize=(8, 6))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
        font_size=12, font_weight='bold', arrows=True, 
        arrowsize=20, edge_color='gray')

# Add to output_vars to display it
output_vars['fig_dag'] = plt.gcf()

# Now let's simulate data following this causal structure
# 1. Generate the exposure
exposure = np.random.binomial(1, 0.5, sample_size)

# 2. Generate mediator based on exposure
mediator = mediation_strength * exposure + np.random.normal(0, 1, sample_size)

# 3. Generate outcome based on both exposure and mediator
outcome = (direct_effect * exposure) + (indirect_effect * mediator) + np.random.normal(0, 1, sample_size)

# Calculate the true total effect
total_effect = direct_effect + indirect_effect * mediation_strength

# Create a dataframe
data = pd.DataFrame({
    'Exposure': exposure,
    'Mediator': mediator,
    'Outcome': outcome
})

# Calculate associations
# Total effect (not controlling for mediator)
import statsmodels.api as sm
X_total = sm.add_constant(data[['Exposure']])
model_total = sm.OLS(data['Outcome'], X_total).fit()
estimated_total_effect = model_total.params['Exposure']

# Direct effect (controlling for mediator)
X_direct = sm.add_constant(data[['Exposure', 'Mediator']])
model_direct = sm.OLS(data['Outcome'], X_direct).fit()
estimated_direct_effect = model_direct.params['Exposure']

# Indirect effect (difference between total and direct)
estimated_indirect_effect = estimated_total_effect - estimated_direct_effect

# Print results
print("Mediation Scenario Results:")
print("-" * 40)
print(f"True direct effect: {direct_effect:.3f}")
print(f"True indirect effect: {indirect_effect * mediation_strength:.3f}")
print(f"True total effect: {total_effect:.3f}")
print("\\nEstimated effects:")
print(f"Estimated total effect: {estimated_total_effect:.3f}")
print(f"Estimated direct effect: {estimated_direct_effect:.3f}")
print(f"Estimated indirect effect: {estimated_indirect_effect:.3f}")

# Create visualizations
# 1. Bar chart of different effects
fig1 = plt.figure(figsize=(10, 6))
effects = ['Direct Effect', 'Indirect Effect', 'Total Effect']
true_values = [direct_effect, indirect_effect * mediation_strength, total_effect]
estimated_values = [estimated_direct_effect, estimated_indirect_effect, estimated_total_effect]

x = np.arange(len(effects))
width = 0.35

plt.bar(x - width/2, true_values, width, label='True', color='blue', alpha=0.7)
plt.bar(x + width/2, estimated_values, width, label='Estimated', color='red', alpha=0.7)

plt.xlabel('Effect Type')
plt.ylabel('Effect Size')
plt.title('Comparison of Direct, Indirect, and Total Effects')
plt.xticks(x, effects)
plt.legend()
output_vars['fig1'] = fig1

# 2. Scatter plot showing exposure-mediator-outcome relationships
fig2 = plt.figure(figsize=(10, 6))
plt.scatter(mediator, outcome, c=exposure, cmap='coolwarm', alpha=0.6)
plt.title('Relationship Between Mediator and Outcome')
plt.xlabel('Mediator')
plt.ylabel('Outcome')
plt.colorbar(label='Exposure')
output_vars['fig2'] = fig2

# 3. Path diagram with effect sizes
fig3 = plt.figure(figsize=(10, 6))
G2 = nx.DiGraph()

# Add nodes with positions
G2.add_node("Exposure", pos=(-1, 0))
G2.add_node("Mediator", pos=(0, 0))
G2.add_node("Outcome", pos=(1, 0))

# Add edges with labels
G2.add_edge('Exposure', 'Mediator', weight=mediation_strength)
G2.add_edge('Mediator', 'Outcome', weight=indirect_effect)
G2.add_edge('Exposure', 'Outcome', weight=direct_effect)

pos = nx.get_node_attributes(G2, 'pos')
nx.draw(G2, pos, with_labels=True, node_size=3000, node_color='lightblue', 
        font_size=12, font_weight='bold', arrows=True, 
        arrowsize=20, edge_color='gray')

# Add edge labels with effect sizes
edge_labels = {('Exposure', 'Mediator'): f"{mediation_strength:.2f}",
               ('Mediator', 'Outcome'): f"{indirect_effect:.2f}",
               ('Exposure', 'Outcome'): f"{direct_effect:.2f}"}
nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels, font_size=10)

plt.title('Path Diagram with Effect Sizes')
plt.axis('off')
output_vars['fig3'] = fig3
"""
    elif dag_scenario == "Collider":
        initial_code = """# Creating and simulating a Collider DAG
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
exposure_effect = 0.5         # Effect of exposure on collider
outcome_effect = 0.5          # Effect of outcome on collider
true_association = 0.0        # True association between exposure and outcome (0 = no association)
sample_size = 1000            # Number of observations

# Set random seed for reproducibility
np.random.seed(42)

# Create DAG using NetworkX
G = nx.DiGraph()

# Add nodes
G.add_node("Exposure", pos=(-1, 1))
G.add_node("Outcome", pos=(1, 1))
G.add_node("Collider", pos=(0, 0))

# Add edges (arrows showing causation)
G.add_edges_from([
    ('Exposure', 'Collider'),   # Exposure affects collider
    ('Outcome', 'Collider')     # Outcome affects collider
])

# Add an edge between exposure and outcome if specified
if true_association != 0:
    G.add_edge('Exposure', 'Outcome')
    
# Draw the DAG
plt.figure(figsize=(8, 6))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
        font_size=12, font_weight='bold', arrows=True, 
        arrowsize=20, edge_color='gray')

# Add to output_vars to display it
output_vars['fig_dag'] = plt.gcf()

# Now let's simulate data following this causal structure
# 1. Generate exposure
exposure = np.random.binomial(1, 0.5, sample_size)

# 2. Generate outcome - by default independent of exposure
if true_association == 0:
    outcome = np.random.normal(0, 1, sample_size)
else:
    # If we want a true association
    outcome = true_association * exposure + np.random.normal(0, 1, sample_size)

# 3. Generate collider based on both exposure and outcome
collider = (exposure_effect * exposure) + (outcome_effect * outcome) + np.random.normal(0, 1, sample_size)

# Create a dataframe
data = pd.DataFrame({
    'Exposure': exposure,
    'Outcome': outcome,
    'Collider': collider
})

# Calculate associations
# Unadjusted association (not controlling for collider)
import statsmodels.api as sm
X_unadj = sm.add_constant(data[['Exposure']])
model_unadj = sm.OLS(data['Outcome'], X_unadj).fit()
unadjusted_association = model_unadj.params['Exposure']

# Adjusted association (controlling for collider)
X_adj = sm.add_constant(data[['Exposure', 'Collider']])
model_adj = sm.OLS(data['Outcome'], X_adj).fit()
adjusted_association = model_adj.params['Exposure']

# Print results
print("Collider Bias Scenario Results:")
print("-" * 40)
print(f"True association between exposure and outcome: {true_association:.3f}")
print(f"Unadjusted association: {unadjusted_association:.3f}")
print(f"Association after adjusting for collider: {adjusted_association:.3f}")

if abs(adjusted_association) > abs(unadjusted_association):
    print("\\nAdjusting for the collider created a biased association!")
    print("This demonstrates collider bias - how conditioning on a common effect")
    print("can create a spurious association between its causes.")
else:
    print("\\nThe adjustment didn't create the expected collider bias.")
    print("Try increasing the effects on the collider or changing other parameters.")

# Stratify the data by collider (low, medium, high)
data['Collider_Strata'] = pd.qcut(data['Collider'], 3, labels=['Low', 'Medium', 'High'])

# Calculate the exposure-outcome association in each stratum
stratum_associations = []
for stratum in ['Low', 'Medium', 'High']:
    stratum_data = data[data['Collider_Strata'] == stratum]
    X_strat = sm.add_constant(stratum_data[['Exposure']])
    model_strat = sm.OLS(stratum_data['Outcome'], X_strat).fit()
    stratum_association = model_strat.params['Exposure']
    stratum_associations.append({'Stratum': stratum, 'Association': stratum_association})

stratum_df = pd.DataFrame(stratum_associations)

print("\\nStratified Analysis (Exposure-Outcome Association by Collider Level):")
for _, row in stratum_df.iterrows():
    print(f"{row['Stratum']} collider: {row['Association']:.3f}")

# Create visualizations
# 1. Scatter plot of exposure vs outcome, overall
fig1 = plt.figure(figsize=(10, 6))
plt.scatter(data['Exposure'] + np.random.normal(0, 0.05, sample_size), # Add jitter for better visualization
            data['Outcome'], 
            alpha=0.6)
plt.title('Exposure vs Outcome (Unadjusted)')
plt.xlabel('Exposure')
plt.ylabel('Outcome')
output_vars['fig1'] = fig1

# 2. Scatter plot of exposure vs outcome, stratified by collider
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, stratum in enumerate(['Low', 'Medium', 'High']):
    stratum_data = data[data['Collider_Strata'] == stratum]
    axes[i].scatter(stratum_data['Exposure'] + np.random.normal(0, 0.05, len(stratum_data)),
                  stratum_data['Outcome'], 
                  alpha=0.6)
    axes[i].set_title(f'Collider Level: {stratum}')
    axes[i].set_xlabel('Exposure')
    if i == 0:
        axes[i].set_ylabel('Outcome')

plt.tight_layout()
output_vars['fig2'] = fig2

# 3. Bar chart of associations
fig3 = plt.figure(figsize=(10, 6))
bars = ['True', 'Unadjusted', 'Adjusted for Collider'] + [f'{s} Collider' for s in ['Low', 'Medium', 'High']]
values = [true_association, unadjusted_association, adjusted_association] + list(stratum_df['Association'])

plt.bar(bars, values)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Exposure-Outcome Association Under Different Conditions')
plt.ylabel('Association Strength')
plt.xticks(rotation=45)
plt.tight_layout()
output_vars['fig3'] = fig3
"""
    elif dag_scenario == "M-Bias":
        initial_code = """# Creating and simulating an M-Bias DAG
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
u1_effect = 0.5         # Effect of U1 on exposure and M
u2_effect = 0.5         # Effect of U2 on outcome and M
causal_effect = 0.3     # True causal effect of exposure on outcome
sample_size = 1000      # Number of observations

# Set random seed for reproducibility
np.random.seed(42)

# Create DAG using NetworkX
G = nx.DiGraph()

# Add nodes
G.add_node("U1", pos=(-1, 1))
G.add_node("U2", pos=(1, 1))
G.add_node("Exposure", pos=(-1, 0))
G.add_node("Outcome", pos=(1, 0))
G.add_node("M", pos=(0, 0.5))

# Add edges (arrows showing causation)
G.add_edges_from([
    ('U1', 'Exposure'),
    ('U1', 'M'),
    ('U2', 'Outcome'),
    ('U2', 'M'),
    ('Exposure', 'Outcome')
])

# Draw the DAG
plt.figure(figsize=(8, 6))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
        font_size=12, font_weight='bold', arrows=True, 
        arrowsize=20, edge_color='gray')

# Add to output_vars to display it
output_vars['fig_dag'] = plt.gcf()

# Now let's simulate data following this causal structure
# 1. Generate unobserved confounders U1 and U2
u1 = np.random.normal(0, 1, sample_size)
u2 = np.random.normal(0, 1, sample_size)

# 2. Generate exposure based on U1
exposure_prob = 1 / (1 + np.exp(-(u1_effect * u1)))
exposure = np.random.binomial(1, exposure_prob, sample_size)

# 3. Generate M based on U1 and U2
m = (u1_effect * u1) + (u2_effect * u2) + np.random.normal(0, 1, sample_size)

# 4. Generate outcome based on exposure and U2
outcome = (causal_effect * exposure) + (u2_effect * u2) + np.random.normal(0, 1, sample_size)

# Create a dataframe
data = pd.DataFrame({
    'U1': u1,  # In reality, U1 and U2 would be unobserved
    'U2': u2,  # We include them here for illustration
    'Exposure': exposure,
    'M': m,
    'Outcome': outcome
})

# Calculate associations
# Unadjusted association
import statsmodels.api as sm
X_unadj = sm.add_constant(data[['Exposure']])
model_unadj = sm.OLS(data['Outcome'], X_unadj).fit()
unadjusted_association = model_unadj.params['Exposure']

# Adjusted for M (can create bias)
X_adj_m = sm.add_constant(data[['Exposure', 'M']])
model_adj_m = sm.OLS(data['Outcome'], X_adj_m).fit()
m_adjusted_association = model_adj_m.params['Exposure']

# Fully adjusted (if we could observe U1 and U2)
X_adj_full = sm.add_constant(data[['Exposure', 'U1', 'U2']])
model_adj_full = sm.OLS(data['Outcome'], X_adj_full).fit()
fully_adjusted_association = model_adj_full.params['Exposure']

# Print results
print("M-Bias Scenario Results:")
print("-" * 40)
print(f"True causal effect: {causal_effect:.3f}")
print(f"Unadjusted association: {unadjusted_association:.3f}")
print(f"Association adjusted for M: {m_adjusted_association:.3f}")
print(f"Fully adjusted association: {fully_adjusted_association:.3f}")

# Determine which estimate is closest to truth
unadj_bias = abs(unadjusted_association - causal_effect)
m_adj_bias = abs(m_adjusted_association - causal_effect)
full_adj_bias = abs(fully_adjusted_association - causal_effect)

print("\\nBias in each estimate:")
print(f"Unadjusted bias: {unadj_bias:.3f}")
print(f"M-adjusted bias: {m_adj_bias:.3f}")
print(f"Fully adjusted bias: {full_adj_bias:.3f}")

if m_adj_bias > unadj_bias:
    print("\\nAdjusting for M increased bias!")
    print("This demonstrates M-bias - when adjusting for a variable creates a")
    print("biased estimate because it opens a backdoor path.")
else:
    print("\\nThe expected M-bias pattern isn't clearly visible.")
    print("Try adjusting the parameter values to make the effect more pronounced.")

# Create visualizations
# 1. Bar chart of the different estimates
fig1 = plt.figure(figsize=(10, 6))
estimates = ['True Causal Effect', 'Unadjusted', 'Adjusted for M', 'Fully Adjusted']
values = [causal_effect, unadjusted_association, m_adjusted_association, fully_adjusted_association]

plt.bar(estimates, values)
plt.axhline(y=causal_effect, color='r', linestyle='--', label='True Effect')
plt.title('Comparison of Causal Effect Estimates')
plt.ylabel('Effect Estimate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
output_vars['fig1'] = fig1

# 2. Scatter plot showing relationships
fig2, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: M vs Exposure colored by U1
axes[0].scatter(data['Exposure'] + np.random.normal(0, 0.05, sample_size), # Add jitter for better visualization
                data['M'], 
                c=data['U1'], 
                cmap='viridis', 
                alpha=0.6)
axes[0].set_title('M vs Exposure (colored by U1)')
axes[0].set_xlabel('Exposure')
axes[0].set_ylabel('M')

# Plot 2: M vs Outcome colored by U2
axes[1].scatter(data['Outcome'], 
                data['M'], 
                c=data['U2'], 
                cmap='viridis', 
                alpha=0.6)
axes[1].set_title('M vs Outcome (colored by U2)')
axes[1].set_xlabel('Outcome')
axes[1].set_ylabel('M')

plt.tight_layout()
output_vars['fig2'] = fig2

# 3. Comparison of bias
fig3 = plt.figure(figsize=(10, 6))
bias_labels = ['Unadjusted Bias', 'M-adjusted Bias', 'Fully Adjusted Bias']
bias_values = [unadj_bias, m_adj_bias, full_adj_bias]

plt.bar(bias_labels, bias_values)
plt.title('Bias in Different Estimation Approaches')
plt.ylabel('Absolute Bias')
plt.tight_layout()
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
            if dag_scenario == "Confounding":
                st.markdown("""
                **Try modifying:**
                - Change `confounding_strength` to see how strong the confounder's effect is
                - Modify `causal_effect` to see the true relationship between exposure and outcome
                - Adjust `sample_size` to see how it affects the precision of estimates
                
                **Simple Challenges:**
                1. Create a setting where confounding completely masks the true effect
                2. Make confounding reverse the direction of the association
                3. Find the minimum sample size needed to detect the true effect
                4. Compare stratification and regression adjustment approaches
                """)
            elif dag_scenario == "Mediation":
                st.markdown("""
                **Try modifying:**
                - Change the `direct_effect` and `indirect_effect` to see how they contribute to the total effect
                - Adjust `mediation_strength` to make the mediator more or less important
                - Try making the direct effect negative and the indirect effect positive
                
                **Simple Challenges:**
                1. Create a scenario where the direct and indirect effects cancel each other out
                2. Make the indirect effect larger than the direct effect
                3. Make the mediation account for 75% of the total effect
                4. See what happens when the mediation strength is very low
                """)
            elif dag_scenario == "Collider":
                st.markdown("""
                **Try modifying:**
                - Change `exposure_effect` and `outcome_effect` to see how strongly they influence the collider
                - Set `true_association` to non-zero to add a true relationship
                - Modify how we stratify by the collider
                
                **Simple Challenges:**
                1. Make the collider-induced bias stronger by increasing effects on the collider
                2. Add a true association and see if adjusting for the collider still creates bias
                3. See what happens when only one variable affects the collider
                4. Create a scenario where collider stratification completely obscures a true effect
                """)
            elif dag_scenario == "M-Bias":
                st.markdown("""
                **Try modifying:**
                - Change `u1_effect` and `u2_effect` to control how strongly they affect their children
                - Modify `causal_effect` to see the true relationship
                - Explore different adjustment strategies
                
                **Simple Challenges:**
                1. Make M-bias stronger by increasing the effects of U1 and U2
                2. See if you can create a scenario where adjusting for M makes the estimate worse than unadjusted
                3. See what happens if the true effect is zero
                4. Try extreme values for the effects to understand the pattern
                """)
    
    if execute_button:
        # Execute the code and display results
        success, output, output_vars = execute_code(user_code)
        
        if success:
            # Display any printed output
            if output:
                st.subheader("Output:")
                st.text(output)
            
            # Display the DAG
            if 'fig_dag' in output_vars:
                st.subheader("Directed Acyclic Graph (DAG):")
                st.pyplot(output_vars['fig_dag'])
            
            # Display other figures
            for i in range(1, 4):  # Check for fig1, fig2, fig3
                fig_key = f'fig{i}'
                if fig_key in output_vars:
                    st.pyplot(output_vars[fig_key])
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section - simplified for beginners
    st.subheader("Key Concepts")
    
    if dag_scenario == "Confounding":
        st.markdown("""
        ### Understanding Confounding:
        
        1. **What is Confounding?**
           - A confounder affects both the exposure and the outcome
           - Creates a "backdoor path" between exposure and outcome
           - Makes the crude (unadjusted) association different from the causal effect
        
        2. **Identifying Confounders:**
           - Must be associated with both exposure and outcome
           - Must not be on the causal pathway (not a mediator)
           - Often represented by arrows going from the confounder to both exposure and outcome
        
        3. **Controlling for Confounding:**
           - Stratification: Analyze the association within levels of the confounder
           - Regression adjustment: Include confounder in statistical model
           - Matching: Create comparable groups with similar confounder distributions
           - Randomization: In experiments, randomly assigns exposure to break confounder-exposure link
        
        4. **When You Don't Control for Confounding:**
           - The measured association will be biased
           - May overestimate or underestimate the true causal effect
           - Can even reverse the direction of the association
        """)
    elif dag_scenario == "Mediation":
        st.markdown("""
        ### Understanding Mediation:
        
        1. **What is Mediation?**
           - Occurs when a variable (mediator) lies on the causal pathway between exposure and outcome
           - The exposure affects the mediator, which then affects the outcome
           - Divides the causal effect into direct and indirect components
        
        2. **Key Measures in Mediation:**
           - **Direct Effect**: Impact of exposure on outcome not through the mediator
           - **Indirect Effect**: Impact of exposure that works through the mediator
           - **Total Effect**: Sum of direct and indirect effects
        
        3. **Analysis Approaches:**
           - To measure total effect: Don't adjust for the mediator
           - To measure direct effect: Adjust for the mediator
           - To measure indirect effect: Calculate the difference between total and direct effects
        
        4. **Practical Implications:**
           - Helps understand mechanisms of how exposures work
           - Can identify intervention targets
           - Important for developing comprehensive prevention strategies
        """)
    elif dag_scenario == "Collider":
        st.markdown("""
        ### Understanding Collider Bias:
        
        1. **What is a Collider?**
           - A variable that is affected by both the exposure and the outcome
           - Or more generally, any variable affected by two other variables in the DAG
           - Represented by arrows going into it from multiple variables
        
        2. **How Collider Bias Works:**
           - Controlling for a collider can create a spurious association between its causes
           - Selection based on a collider can distort relationships
           - Creates association where none exists, or distorts a true association
        
        3. **Common Examples:**
           - **Selection Bias**: Study participation affected by multiple factors
           - **Berkson's Fallacy**: Hospital-based studies where multiple conditions affect admission
           - **Survival Bias**: Only analyzing subjects who survive (survival affected by multiple factors)
        
        4. **Avoiding Collider Bias:**
           - Don't adjust for variables affected by both exposure and outcome
           - Be cautious about selection criteria in study design
           - Be skeptical of associations found only in specific subgroups
        """)
    elif dag_scenario == "M-Bias":
        st.markdown("""
        ### Understanding M-Bias:
        
        1. **What is M-Bias?**
           - A specific pattern of bias named for the shape it creates in a DAG
           - Occurs when adjusting for a variable opens a backdoor path
           - The variable (M) is a collider between unmeasured causes of exposure and outcome
        
        2. **The M-Bias Structure:**
           - Unmeasured variable U1 affects both Exposure and M
           - Unmeasured variable U2 affects both Outcome and M
           - Adjusting for M opens a backdoor path: Exposure ← U1 → M ← U2 → Outcome
        
        3. **Practical Implications:**
           - Shows that adjusting for a variable can sometimes increase bias
           - Demonstrates that more adjustment isn't always better
           - Illustrates why causal diagrams are important for identifying proper adjustment sets
        
        4. **Addressing M-Bias:**
           - Use DAGs to identify appropriate adjustment sets
           - Don't automatically adjust for all measured variables
           - Consider the causal structure when designing analyses
        """)

def causal_effects_estimation_lesson():
    st.header("Causal Effects Estimation")
    
    st.markdown("""
    ### Estimating Causal Effects
    
    This exercise explores how to estimate causal effects from observational data.
    We'll simulate a realistic scenario and apply different causal inference methods.
    """)
    
    # Initial code example - simplified for beginners
    initial_code = """# Causal effects estimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic

# MODIFY THESE VALUES TO SEE WHAT HAPPENS
sample_size = 1000        # Number of subjects
true_effect = 0.5         # True causal effect of treatment
confounding_strength = 0.7  # How strongly confounders affect treatment and outcome
n_confounders = 2         # Number of confounders

# Set random seed for reproducibility
np.random.seed(42)

# Generate data
# 1. Generate confounders
confounders = np.random.normal(0, 1, (sample_size, n_confounders))

# 2. Generate propensity score (probability of treatment)
# Higher values of confounders increase probability of treatment
propensity_logit = np.sum(confounding_strength * confounders, axis=1)
propensity = 1 / (1 + np.exp(-propensity_logit))

# 3. Assign treatment based on propensity
treatment = np.random.binomial(1, propensity, sample_size)

# 4. Generate outcome
# Outcome depends on treatment and confounders
outcome = (true_effect * treatment + 
           np.sum(confounding_strength * confounders, axis=1) + 
           np.random.normal(0, 1, sample_size))

# Create a DataFrame
confounder_cols = [f'Confounder_{i+1}' for i in range(n_confounders)]
data = pd.DataFrame(confounders, columns=confounder_cols)
data['Treatment'] = treatment
data['Outcome'] = outcome
data['Propensity'] = propensity

print("Generated Dataset:")
print(data.head())
print("\\nData Summary:")
print(data.describe().round(2))

# 1. Naive estimate (ignoring confounding)
naive_effect = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])

# 2. Regression adjustment
# Include confounders in the regression model
formula = f"Outcome ~ Treatment + {' + '.join(confounder_cols)}"
reg_model = smf.ols(formula, data=data).fit()
regression_effect = reg_model.params['Treatment']

# 3. Stratification
# Create strata based on propensity score quintiles
data['Propensity_Strata'] = pd.qcut(data['Propensity'], 5, labels=False)

# Calculate treatment effect within each stratum
strata_effects = []
for stratum in range(5):
    stratum_data = data[data['Propensity_Strata'] == stratum]
    n_stratum = len(stratum_data)
    n_treated = sum(stratum_data['Treatment'])
    n_control = n_stratum - n_treated
    
    # Skip strata with too few in either group
    if n_treated < 5 or n_control < 5:
        continue
    
    # Calculate effect in this stratum
    stratum_effect = np.mean(stratum_data[stratum_data['Treatment'] == 1]['Outcome']) - np.mean(stratum_data[stratum_data['Treatment'] == 0]['Outcome'])
    stratum_weight = n_stratum / sample_size
    strata_effects.append((stratum, stratum_effect, stratum_weight, n_stratum, n_treated, n_control))

# Calculate weighted average of stratum-specific effects
stratification_effect = sum(effect * weight for _, effect, weight, _, _, _ in strata_effects) / sum(weight for _, _, weight, _, _, _ in strata_effects)

# 4. Inverse Probability Weighting (IPW)
# Create weights
data['IPW_Weight'] = treatment / propensity + (1 - treatment) / (1 - propensity)

# Trim extreme weights
weight_percentile_99 = np.percentile(data['IPW_Weight'], 99)
data['IPW_Weight_Trimmed'] = np.minimum(data['IPW_Weight'], weight_percentile_99)

# Weighted outcome model
weighted_model = smf.wls("Outcome ~ Treatment", data=data, weights=data['IPW_Weight_Trimmed']).fit()
ipw_effect = weighted_model.params['Treatment']

# Print results
print("\\nCausal Effect Estimates:")
print(f"True Effect: {true_effect:.3f}")
print(f"Naive Estimate (ignoring confounding): {naive_effect:.3f}")
print(f"Regression Adjustment Estimate: {regression_effect:.3f}")
print(f"Stratification Estimate: {stratification_effect:.3f}")
print(f"Inverse Probability Weighting Estimate: {ipw_effect:.3f}")

# Calculate bias for each method
naive_bias = abs(naive_effect - true_effect)
regression_bias = abs(regression_effect - true_effect)
stratification_bias = abs(stratification_effect - true_effect)
ipw_bias = abs(ipw_effect - true_effect)

print("\\nBias in each estimate:")
print(f"Naive Estimate Bias: {naive_bias:.3f}")
print(f"Regression Adjustment Bias: {regression_bias:.3f}")
print(f"Stratification Bias: {stratification_bias:.3f}")
print(f"IPW Bias: {ipw_bias:.3f}")

# Create visualizations

# 1. Bar chart of effect estimates
fig1 = plt.figure(figsize=(10, 6))
estimates = ['True Effect', 'Naive', 'Regression', 'Stratification', 'IPW']
values = [true_effect, naive_effect, regression_effect, stratification_effect, ipw_effect]

plt.bar(estimates, values)
plt.axhline(y=true_effect, color='r', linestyle='--', label='True Effect')
plt.title('Comparison of Causal Effect Estimates')
plt.ylabel('Effect Estimate')
plt.legend()
output_vars['fig1'] = fig1

# 2. Propensity score distribution
fig2 = plt.figure(figsize=(10, 6))
plt.hist(data[data['Treatment'] == 1]['Propensity'], alpha=0.5, bins=20, label='Treated')
plt.hist(data[data['Treatment'] == 0]['Propensity'], alpha=0.5, bins=20, label='Control')
plt.title('Propensity Score Distribution by Treatment Status')
plt.xlabel('Propensity Score')
plt.ylabel('Count')
plt.legend()
output_vars['fig2'] = fig2

# 3. Stratification results
fig3 = plt.figure(figsize=(10, 6))

# Extract data for plotting
strata = [s[0] for s in strata_effects]
effects = [s[1] for s in strata_effects]
weights = [s[2] for s in strata_effects]
sizes = [s[3] for s in strata_effects]

# Plot stratum-specific effects
plt.bar(strata, effects, width=0.4)
plt.axhline(y=true_effect, color='r', linestyle='--', label='True Effect')
plt.axhline(y=stratification_effect, color='g', linestyle='--', label='Weighted Average')
plt.title('Effect Estimates by Propensity Score Stratum')
plt.xlabel('Propensity Score Stratum')
plt.ylabel('Treatment Effect')
plt.xticks(strata)
plt.legend()

# Add stratum sizes as text
for i, (stratum, effect, weight, size, n_treated, n_control) in enumerate(strata_effects):
    plt.text(stratum, effect, f"n={size}\\n({n_treated}T/{n_control}C)", 
             ha='center', va='bottom')

output_vars['fig3'] = fig3

# 4. Bias comparison
fig4 = plt.figure(figsize=(10, 6))
methods = ['Naive', 'Regression', 'Stratification', 'IPW']
biases = [naive_bias, regression_bias, stratification_bias, ipw_bias]

plt.bar(methods, biases)
plt.title('Bias in Different Estimation Methods')
plt.ylabel('Absolute Bias')
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
        # Add hints and challenges - simplified for beginners
        with st.expander("Ideas to Try"):
            st.markdown("""
            **Try modifying:**
            - Change `true_effect` to simulate different treatment effects
            - Adjust `confounding_strength` to control how much bias is present
            - Modify `n_confounders` to add more confounding variables
            - Change `sample_size` to see how it affects estimation precision
            
            **Simple Challenges:**
            1. Create a scenario with strong confounding that completely masks the true effect
            2. Compare how different methods perform when sample size is small
            3. See how adding more confounders affects the different estimation methods
            4. Create a situation where the naive estimate suggests harm but the true effect is beneficial
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
            for i in range(1, 5):  # Check for fig1, fig2, fig3, fig4
                fig_key = f'fig{i}'
                if fig_key in output_vars:
                    st.pyplot(output_vars[fig_key])
        else:
            # Display error message
            st.error("Error in your code:")
            st.code(output)
    
    # Include a discussion section - simplified for beginners
    st.subheader("Key Concepts")
    st.markdown("""
    ### Causal Effect Estimation Methods:
    
    1. **Naive Estimate**:
       - Simply compares treated and untreated groups
       - Doesn't account for confounding
       - Usually biased in observational studies
       - Only valid when treatment is randomly assigned
    
    2. **Regression Adjustment**:
       - Includes confounders as covariates in a regression model
       - Controls for measured confounders
       - Assumes correct model specification
       - Easy to implement but sensitive to model form
    
    3. **Propensity Score Methods**:
       - **Propensity Score**: Probability of treatment given confounders
       - **Stratification**: Group by similar propensity scores and compare within strata
       - **Matching**: Match treated subjects to controls with similar propensity scores
       - **Weighting**: Weight observations by inverse probability of treatment received
       - Helps balance confounders between treated and untreated groups
    
    4. **Inverse Probability Weighting (IPW)**:
       - Creates a pseudo-population where treatment is independent of confounders
       - Upweights underrepresented combinations of treatment and confounders
       - Sensitive to extreme weights (weight trimming often needed)
       - Provides doubly-robust estimation when combined with outcome regression
    
    5. **Key Assumptions for All Methods**:
       - **No unmeasured confounding**: All important confounders are measured
       - **Positivity**: Each subject has non-zero probability of each treatment
       - **Consistency**: Well-defined treatment with consistent effect
       - **Correct implementation**: Model specifications, weights, matches are appropriate
    """)

if __name__ == "__main__":
    app()