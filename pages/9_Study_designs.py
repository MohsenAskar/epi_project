# pages/7_study_designs.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time
from lifelines import KaplanMeierFitter
import time as time_module
from scipy.stats import norm


# Add path to import the code lab module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interactive_code'))
import study_designs_code

def safe_annotation_position(base_x, base_y, text_length, nearby_elements, buffer=20):
    """Calculate safe position for annotations avoiding overlap"""
    # Adjust x based on text length
    adjusted_x = base_x
    # Adjust y to avoid nearby elements
    adjusted_y = base_y
    for elem_x, elem_y in nearby_elements:
        if abs(adjusted_x - elem_x) < buffer and abs(adjusted_y - elem_y) < buffer:
            adjusted_y += buffer
    return adjusted_x, adjusted_y

st.set_page_config(layout="wide")

st.title("Epidemiological Study Designs")

viz_tab, code_tab = st.tabs(["📊 Interactive Visualization", "💻 Code Laboratory"])

# Wrap your existing visualization content in:
with viz_tab:
    # Study design selector
    study_design = st.selectbox(
        "Select Study Design",
        ["Cohort Study", "Case-Control Study", "Randomized Controlled Trial", 
        "Cross-sectional Study", "Nested Case-Control Study", "Case-Cohort Study"]
    )

    # ENHANCED COHORT STUDY SECTION
    if study_design == "Cohort Study":
        st.header("Cohort Study Simulation")
        
        # Create tabs for different aspects of the visualization
        tab_main, tab_concepts, tab_interactive, tab_compare = st.tabs([
            "📉 Main Visualization", 
            "📝 Conceptual Overview", 
            "📊 Interactive Exploration", 
            "⚖️ Design Comparison"
        ])
        
        # Parameters for cohort study (shared across all tabs)
        with tab_main:
            col1, col2 = st.columns(2)
            with col1:
                n_subjects = st.slider("Number of Subjects", 100, 1000, 500, key="cohort_size")
                followup_years = st.slider("Follow-up Years", 1, 10, 5, key="cohort_followup")
            with col2:
                baseline_risk = st.slider("Baseline Risk per Year", 0.01, 0.10, 0.05, key="cohort_risk")
                relative_risk = st.slider("Relative Risk for Exposed", 1.0, 5.0, 2.0, key="cohort_effect")
        
        # Generate cohort data
        np.random.seed(42)
        exposed = np.random.binomial(1, 0.5, n_subjects)
        years_to_event = np.array([
            np.random.exponential(1 / (baseline_risk * (relative_risk if e else 1)))
            for e in exposed
        ])
        censored = years_to_event > followup_years
        years_to_event[censored] = followup_years
        
        cohort_data = pd.DataFrame({
            'Subject_ID': range(1, n_subjects + 1),
            'Exposed': exposed,
            'Years_to_Event': years_to_event,
            'Censored': censored,
            'Event_Occurred': ~censored
        })
        
        # Main visualization tab
        with tab_main:
            # Step-by-step progression slider
            steps = st.radio(
                "Study Design Stages:",
                ["1. Define Cohort", "2. Assess Exposure", "3. Follow Over Time", 
                "4. Measure Outcomes", "5. Analyze Results"],
                horizontal=True,
                key="cohort_steps"
            )
            
            # Create dynamic visualization based on selected step
            fig = go.Figure()
            
            # Constants for visualization
            total_width = 800
            total_height = 600
            timeline_y = 350
            
            # Different elements based on step
            if steps == "1. Define Cohort":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=350 - 120, x1=350 + 120,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=350, y=timeline_y + 100,
                    text=f"Study Population<br>n={n_subjects}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=550,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=350, y=timeline_y - 100,
                    text="Step 1: Define and recruit a cohort of individuals<br>" +
                        f"• Cohort size: {n_subjects} participants<br>" +
                        "• Participants are free of disease at baseline<br>" +
                        "• Collect baseline information",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
            elif steps == "2. Assess Exposure":
                # Draw timeline
                fig.add_shape(
                    type="line",
                    x0=100, x1=600,
                    y0=timeline_y, y1=timeline_y,
                    line=dict(color="black", width=2),
                )
                
                fig.add_annotation(
                    x=100, y=timeline_y - 20,
                    text="Baseline",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Split into exposed and unexposed groups
                exposed_count = sum(exposed)
                unexposed_count = n_subjects - exposed_count
                
                # Draw exposed group
                fig.add_shape(
                    type="rect",
                    x0=250 - 80, x1=250 + 80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="orange", width=2),
                    fillcolor="rgba(255, 165, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=250, y=timeline_y + 100,
                    text=f"Exposed Group<br>n={exposed_count}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw unexposed group
                fig.add_shape(
                    type="rect",
                    x0=450 - 80, x1=450 + 80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0, 0, 255, 0.2)"
                )
                
                fig.add_annotation(
                    x=450, y=timeline_y + 100,
                    text=f"Unexposed Group<br>n={unexposed_count}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=550,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=350, y=timeline_y - 100,
                    text="Step 2: Assess exposure status at baseline<br>" +
                        f"• Exposed group: {exposed_count} participants<br>" +
                        f"• Unexposed group: {unexposed_count} participants<br>" +
                        "• Groups should be comparable except for exposure",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
            elif steps == "3. Follow Over Time":
                # Draw timeline
                fig.add_shape(
                    type="line",
                    x0=100, x1=600,
                    y0=timeline_y, y1=timeline_y,
                    line=dict(color="black", width=2),
                )
                
                # Add time labels
                for year in range(followup_years + 1):
                    x_pos = 100 + year * (500 / followup_years)
                    fig.add_annotation(
                        x=x_pos,
                        y=timeline_y - 20,
                        text=f"Year {year}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                
                # Sample participants for visualization
                n_sample = 20
                sample_data = cohort_data.sample(n_sample)
                
                # Split into exposed and unexposed
                exposed_sample = sample_data[sample_data['Exposed'] == 1]
                unexposed_sample = sample_data[sample_data['Exposed'] == 0]
                
                # Draw follow-up lines for exposed group
                for i, (_, subject) in enumerate(exposed_sample.iterrows()):
                    y_pos = timeline_y + 100 - i * 10
                    x_end = 100 + (subject['Years_to_Event'] / followup_years) * 500
                    
                    fig.add_shape(
                        type="line",
                        x0=100, x1=x_end,
                        y0=y_pos, y1=y_pos,
                        line=dict(color="orange", width=1.5),
                    )
                    
                    # Add event marker if applicable
                    if subject['Event_Occurred']:
                        fig.add_trace(go.Scatter(
                            x=[x_end],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(color="red", size=8, symbol="x"),
                            showlegend=False
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=[x_end],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(color="black", size=6, symbol="circle-open"),
                            showlegend=False
                        ))
                
                # Draw follow-up lines for unexposed group
                for i, (_, subject) in enumerate(unexposed_sample.iterrows()):
                    y_pos = timeline_y - 100 + i * 10
                    x_end = 100 + (subject['Years_to_Event'] / followup_years) * 500
                    
                    fig.add_shape(
                        type="line",
                        x0=100, x1=x_end,
                        y0=y_pos, y1=y_pos,
                        line=dict(color="blue", width=1.5),
                    )
                    
                    # Add event marker if applicable
                    if subject['Event_Occurred']:
                        fig.add_trace(go.Scatter(
                            x=[x_end],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(color="red", size=8, symbol="x"),
                            showlegend=False
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=[x_end],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(color="black", size=6, symbol="circle-open"),
                            showlegend=False
                        ))
                
                # Add group labels
                fig.add_annotation(
                    x=80, y=timeline_y + 50,
                    text="Exposed<br>Group",
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="right"
                )
                
                fig.add_annotation(
                    x=80, y=timeline_y - 50,
                    text="Unexposed<br>Group",
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="right"
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=550,
                    y0=timeline_y - 240, y1=timeline_y - 130,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=350, y=timeline_y - 185,
                    text="Step 3: Follow participants over time<br>" +
                        f"• Follow-up duration: {followup_years} years<br>" +
                        "• Monitor for disease occurrence<br>" +
                        "• X marks indicate event occurrence",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
            elif steps == "4. Measure Outcomes":
                # Calculate outcome statistics
                exposed_count = sum(cohort_data['Exposed'])
                unexposed_count = n_subjects - exposed_count
                
                events_exposed = cohort_data[cohort_data['Exposed'] == 1]['Event_Occurred'].sum()
                events_unexposed = cohort_data[cohort_data['Exposed'] == 0]['Event_Occurred'].sum()
                
                risk_exposed = events_exposed / exposed_count
                risk_unexposed = events_unexposed / unexposed_count
                
                # Create outcome visual
                # Draw timeline
                fig.add_shape(
                    type="line",
                    x0=100, x1=600,
                    y0=timeline_y, y1=timeline_y,
                    line=dict(color="black", width=2),
                )
                
                # Add time markers
                fig.add_annotation(
                    x=100, y=timeline_y - 20,
                    text="Baseline",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                fig.add_annotation(
                    x=600, y=timeline_y - 20,
                    text=f"Year {followup_years}",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Create boxes for outcomes
                # Exposed group outcomes
                fig.add_shape(
                    type="rect",
                    x0=250 - 80, x1=250 + 80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="orange", width=2),
                    fillcolor="rgba(255, 165, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=250, y=timeline_y + 100,
                    text=f"Exposed Group<br>Events: {events_exposed}/{exposed_count}<br>Risk: {risk_exposed:.2f}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Unexposed group outcomes
                fig.add_shape(
                    type="rect",
                    x0=450 - 80, x1=450 + 80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0, 0, 255, 0.2)"
                )
                
                fig.add_annotation(
                    x=450, y=timeline_y + 100,
                    text=f"Unexposed Group<br>Events: {events_unexposed}/{unexposed_count}<br>Risk: {risk_unexposed:.2f}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=550,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=350, y=timeline_y - 100,
                    text="Step 4: Measure outcomes<br>" +
                        f"• Events in exposed: {events_exposed} ({events_exposed/exposed_count:.1%})<br>" +
                        f"• Events in unexposed: {events_unexposed} ({events_unexposed/unexposed_count:.1%})<br>" +
                        "• Compare incidence between groups",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
            elif steps == "5. Analyze Results":
                # Calculate outcome statistics
                exposed_count = sum(cohort_data['Exposed'])
                unexposed_count = n_subjects - exposed_count
                
                events_exposed = cohort_data[cohort_data['Exposed'] == 1]['Event_Occurred'].sum()
                events_unexposed = cohort_data[cohort_data['Exposed'] == 0]['Event_Occurred'].sum()
                
                risk_exposed = events_exposed / exposed_count
                risk_unexposed = events_unexposed / unexposed_count
                risk_ratio = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
                risk_difference = risk_exposed - risk_unexposed
                
                # Create analysis visual
                # Results box
                fig.add_shape(
                    type="rect",
                    x0=250, x1=550,
                    y0=timeline_y - 30, y1=timeline_y + 120,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.1)"
                )
                
                # Add result metrics
                results_text = (
                    f"<b>Cohort Study Results</b><br><br>"
                    f"Risk in Exposed: {risk_exposed:.3f}<br>"
                    f"Risk in Unexposed: {risk_unexposed:.3f}<br><br>"
                    f"Risk Ratio (RR): {risk_ratio:.2f}<br>"
                    f"Risk Difference (RD): {risk_difference:.3f}<br>"
                    f"Number Needed to Harm: {abs(1/risk_difference):.1f}"
                )
                
                fig.add_annotation(
                    x=400, y=timeline_y + 50,
                    text=results_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=200, x1=600,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=400, y=timeline_y - 100,
                    text="Step 5: Analyze results<br>" +
                        f"• Risk Ratio (RR): {risk_ratio:.2f} (ratio of risks in exposed vs. unexposed)<br>" +
                        f"• Risk Difference (RD): {risk_difference:.3f} (absolute difference in risk)<br>" +
                        "• A Risk Ratio > 1 suggests exposure increases risk",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
            
            # Add legend
            if steps in ["3. Follow Over Time", "4. Measure Outcomes"]:
                # Add legend items based on step
                legend_items = []
                
                if steps == "3. Follow Over Time":
                    legend_items = [
                        {"name": "Exposed", "color": "orange", "type": "line"},
                        {"name": "Unexposed", "color": "blue", "type": "line"},
                        {"name": "Event", "color": "red", "symbol": "x", "type": "marker"},
                        {"name": "Censored", "color": "black", "symbol": "circle-open", "type": "marker"}
                    ]
                elif steps == "4. Measure Outcomes":
                    legend_items = [
                        {"name": "Exposed Group", "color": "orange", "type": "box"},
                        {"name": "Unexposed Group", "color": "blue", "type": "box"}
                    ]
                
                # Place legend outside main plot
                legend_x = 700
                legend_y_start = timeline_y + 100
                
                for i, item in enumerate(legend_items):
                    y_pos = legend_y_start - i * 30
                    
                    if item["type"] == "line":
                        fig.add_shape(
                            type="line",
                            x0=legend_x - 30, x1=legend_x,
                            y0=y_pos, y1=y_pos,
                            line=dict(color=item["color"], width=2),
                        )
                    elif item["type"] == "marker":
                        fig.add_trace(go.Scatter(
                            x=[legend_x - 15],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(
                                color=item["color"], 
                                size=10, 
                                symbol=item.get("symbol", "circle")
                            ),
                            showlegend=False
                        ))
                    elif item["type"] == "box":
                        fig.add_shape(
                            type="rect",
                            x0=legend_x - 30, x1=legend_x,
                            y0=y_pos - 10, y1=y_pos + 10,
                            line=dict(color=item["color"], width=2),
                            fillcolor=f"rgba({255 if item['color']=='orange' else 0}, {165 if item['color']=='orange' else 0}, {0 if item['color']=='orange' else 255}, 0.2)"
                        )
                    
                    fig.add_annotation(
                        x=legend_x + 40,
                        y=y_pos,
                        text=item["name"],
                        showarrow=False,
                        font=dict(size=12),
                        xanchor="left"
                    )
            
            # Update layout
            fig.update_layout(
                title="Cohort Study Design",
                height=650,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[timeline_y - 250, timeline_y + 200]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Optional animation
            if st.checkbox("Show animated progression through follow-up", key="cohort_animation"):
                progress_bar = st.progress(0)
                current_year = st.empty()
                animation_plot = st.empty()
                animation_description = st.empty()
                # Create animation
                for t in range(0, followup_years * 10 + 1):
                    time_point = t / 10
                    progress = int(100 * time_point / followup_years)
                    progress_bar.progress(progress)
                    current_year.markdown(f"### Year {time_point:.1f}")
                    
                    # Create frame for this time point
                    anim_fig = go.Figure()
                    
                    # Draw timeline
                    anim_fig.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=0, y1=0,
                        line=dict(color="black", width=2),
                    )
                    
                    # Add time marker
                    anim_fig.add_shape(
                        type="line",
                        x0=time_point, x1=time_point,
                        y0=-0.2, y1=20,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    # Add participants (sample)
                    sample_size = 20
                    sample = cohort_data.sample(sample_size)
                    # Count events up to this time point for the animation
                    events_exposed_so_far = sum((sample['Exposed'] == 1) & 
                                            (sample['Event_Occurred'] == 1) & 
                                            (sample['Years_to_Event'] <= time_point))
                    events_unexposed_so_far = sum((sample['Exposed'] == 0) & 
                                                (sample['Event_Occurred'] == 1) & 
                                                (sample['Years_to_Event'] <= time_point))
                    exposed_in_sample = sum(sample['Exposed'] == 1)
                    unexposed_in_sample = sum(sample['Exposed'] == 0)
                    
                    for i, (_, subject) in enumerate(sample.iterrows()):
                        y_pos = i + 1
                        color = "orange" if subject['Exposed'] else "blue"
                        
                        # Draw line up to current time or event/censoring time
                        x_end = min(time_point, subject['Years_to_Event'])
                        
                        anim_fig.add_shape(
                            type="line",
                            x0=0, x1=x_end,
                            y0=y_pos, y1=y_pos,
                            line=dict(color=color, width=1.5),
                        )
                        
                        # Add event marker if applicable
                        if subject['Event_Occurred'] and subject['Years_to_Event'] <= time_point:
                            anim_fig.add_trace(go.Scatter(
                                x=[subject['Years_to_Event']],
                                y=[y_pos],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                name="Event",
                                showlegend=False
                            ))
                    
                    # Update layout
                    anim_fig.update_layout(
                        title=f"Cohort Follow-up at Year {time_point:.1f}",
                        xaxis_title="Follow-up Time (Years)",
                        height=500,
                        xaxis=dict(range=[-0.1, followup_years + 0.1]),
                        yaxis=dict(showticklabels=False, range=[0, sample_size + 1])
                    )
                    
                    animation_plot.plotly_chart(anim_fig, use_container_width=True)
                    with animation_description.container():
                        st.subheader("What's happening in this animation:")
                        
                        st.write(f"**Current time point:** Year {time_point:.1f} of {followup_years} years follow-up")
                        
                        st.write("**What you're seeing:**")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown("• <span style='color:orange; font-weight:bold;'>Orange lines:</span>", unsafe_allow_html=True)
                            st.markdown("• <span style='color:blue; font-weight:bold;'>Blue lines:</span>", unsafe_allow_html=True)
                            st.markdown("• <span style='color:red; font-weight:bold;'>Red X markers:</span>", unsafe_allow_html=True)
                            st.markdown("• <span style='color:red; font-weight:bold;'>Vertical red line:</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.write(f"Participants in the exposed group ({exposed_in_sample} people)")
                            st.write(f"Participants in the unexposed group ({unexposed_in_sample} people)")
                            st.write("Disease events that have occurred")
                            st.write("Current point in follow-up time")
                        
                        st.write(f"**Events so far:** {events_exposed_so_far} in exposed group, {events_unexposed_so_far} in unexposed group")
                        
                        st.write("This animation demonstrates how cohort studies follow participants prospectively over time, "
                                "recording when disease events occur in both exposed and unexposed groups. The difference in "
                                "event occurrence between groups allows us to calculate measures like risk ratio and risk difference.")
                        
                        # Add context-specific message in a different color
                        if time_point < followup_years * 0.25:
                            st.info("**Early follow-up:** Most participants are still event-free. We're just beginning to collect outcome data.")
                        elif time_point < followup_years * 0.75:
                            st.info("**Mid follow-up:** Events are accumulating. We can start to see patterns in which group experiences more events.")
                        else:
                            st.info("**Late follow-up:** Approaching study completion. The final pattern of events will determine our risk estimates.")
                    
                    time_module.sleep(1)  # Control animation speed 
                           
        # Conceptual overview tab
        with tab_concepts:
            st.subheader("Cohort Study: Conceptual Overview")
            
            st.write("""
            A cohort study follows a group of people over time to determine the relationship between exposure and outcome.
            """)
            
            # Create a two-column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Key Strengths")
                st.write("""
                1. **Temporal sequence**: Clearly establishes that exposure precedes outcome
                2. **Multiple outcomes**: Can examine various outcomes from a single exposure
                3. **Incidence**: Directly measures disease incidence and allows calculation of risk
                4. **Rare exposures**: Efficient for studying uncommon exposures
                """)
            
            with col2:
                st.write("#### Limitations")
                st.write("""
                1. **Time and resources**: Often requires large sample sizes and long follow-up
                2. **Loss to follow-up**: Participants may drop out over time
                3. **Inefficient for rare diseases**: Requires large samples to observe enough cases
                4. **Selection bias**: Those who participate may differ from general population
                """)
            
            # Create a simple process flow diagram
            process_fig = go.Figure()
            steps = ["Select Cohort", "Assess Exposure", "Follow Over Time", "Measure Outcomes", "Analyze Results"]
            x_positions = [100, 250, 400, 550, 700]
            y_position = 100
            
            # Draw process boxes
            for i, (step, x_pos) in enumerate(zip(steps, x_positions)):
                # Draw box
                process_fig.add_shape(
                    type="rect",
                    x0=x_pos-70, x1=x_pos+70,
                    y0=y_position-30, y1=y_position+30,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(100, 149, 237, 0.3)"
                )
                
                # Add step label
                process_fig.add_annotation(
                    x=x_pos, y=y_position,
                    text=f"{i+1}. {step}",
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                # Add connecting line (except for the last step)
                if i < len(steps) - 1:
                    next_x = x_positions[i+1]
                    process_fig.add_shape(
                        type="line",
                        x0=x_pos+70, x1=next_x-70,
                        y0=y_position, y1=y_position,
                        line=dict(color="blue", width=1),
                    )
            
            # Update layout
            process_fig.update_layout(
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[50, 150]
                )
            )
            
            st.plotly_chart(process_fig, use_container_width=True)
        
        # Interactive exploration tab
        with tab_interactive:
            st.subheader("Interactive Risk Explorer")
            
            # Let user adjust parameters
            col1, col2 = st.columns(2)
            
            with col1:
                custom_exposure_prev = st.slider(
                    "Exposure Prevalence", 
                    0.1, 0.9, 0.5, 
                    key="custom_exposure_prev"
                )
                custom_baseline_risk = st.slider(
                    "Baseline Risk (Unexposed)", 
                    0.01, 0.5, 0.1, 
                    key="custom_baseline_risk"
                )
            
            with col2:
                custom_rr = st.slider(
                    "Relative Risk (Exposed vs. Unexposed)", 
                    1.0, 10.0, 2.0, 
                    key="custom_rr"
                )
                custom_n = st.slider(
                    "Sample Size", 
                    100, 10000, 1000, 
                    step=100,
                    key="custom_n"
                )
            
            # Calculate expected events
            exposed_n = int(custom_n * custom_exposure_prev)
            unexposed_n = custom_n - exposed_n
            
            exposed_risk = min(custom_baseline_risk * custom_rr, 1.0)
            exposed_events = int(exposed_n * exposed_risk)
            unexposed_events = int(unexposed_n * custom_baseline_risk)
            
            total_events = exposed_events + unexposed_events
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Exposed Group", f"{exposed_n} participants")
                st.metric("Events in Exposed", f"{exposed_events} ({exposed_events/exposed_n:.1%})")
            
            with col2:
                st.metric("Unexposed Group", f"{unexposed_n} participants")
                st.metric("Events in Unexposed", f"{unexposed_events} ({unexposed_events/unexposed_n:.1%})")
            
            with col3:
                custom_rr_observed = (exposed_events/exposed_n)/(unexposed_events/unexposed_n) if unexposed_events > 0 else float('inf')
                custom_rd = (exposed_events/exposed_n) - (unexposed_events/unexposed_n)
                
                st.metric("Observed Relative Risk", f"{custom_rr_observed:.2f}")
                st.metric("Risk Difference", f"{custom_rd:.3f}")
                
                if custom_rd != 0:
                    st.metric("Number Needed to Harm/Treat", f"{abs(1/custom_rd):.1f}")
            
            # Create a visual representation
            fig = go.Figure()
            
            # Create a relative risk visualization
            # First, create a 2x2 table visual
            table_width = 400
            table_height = 300
            cell_width = table_width / 2
            cell_height = table_height / 2
            
            # Draw the table borders
            # Vertical line
            fig.add_shape(
                type="line",
                x0=table_width/2, x1=table_width/2,
                y0=0, y1=table_height,
                line=dict(color="black", width=2),
            )
            
            # Horizontal line
            fig.add_shape(
                type="line",
                x0=0, x1=table_width,
                y0=table_height/2, y1=table_height/2,
                line=dict(color="black", width=2),
            )
            
            # Add labels
            # Column headers
            fig.add_annotation(
                x=cell_width/2, y=table_height + 20,
                text="Exposed",
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.add_annotation(
                x=cell_width + cell_width/2, y=table_height + 20,
                text="Unexposed",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Row headers
            fig.add_annotation(
                x=-50, y=cell_height + cell_height/2,
                text="No Disease",
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.add_annotation(
                x=-50, y=cell_height/2,
                text="Disease",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Add cell values
            # Top left: Exposed, No Disease
            fig.add_annotation(
                x=cell_width/2, y=cell_height + cell_height/2,
                text=f"{exposed_n - exposed_events}",
                showarrow=False,
                font=dict(size=16)
            )
            
            # Top right: Unexposed, No Disease
            fig.add_annotation(
                x=cell_width + cell_width/2, y=cell_height + cell_height/2,
                text=f"{unexposed_n - unexposed_events}",
                showarrow=False,
                font=dict(size=16)
            )
            
            # Bottom left: Exposed, Disease
            fig.add_annotation(
                x=cell_width/2, y=cell_height/2,
                text=f"{exposed_events}",
                showarrow=False,
                font=dict(size=16)
            )
            
            # Bottom right: Unexposed, Disease
            fig.add_annotation(
                x=cell_width + cell_width/2, y=cell_height/2,
                text=f"{unexposed_events}",
                showarrow=False,
                font=dict(size=16)
            )
            
            # Add result annotation
            fig.add_annotation(
                x=table_width/2, y=-50,
                text=f"Relative Risk (RR) = {custom_rr_observed:.2f}<br>Risk Difference (RD) = {custom_rd:.3f}",
                showarrow=False,
                font=dict(size=14),
                align="center"
            )
            
            # Update layout
            fig.update_layout(
                title="Interactive 2×2 Table Visualization",
                height=400,
                width=500,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-100, table_width + 50]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-100, table_height + 50]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a risk curve visualization
            st.subheader("Effect of Relative Risk on Observed Events")
            
            # Create a range of RR values
            rr_range = np.linspace(1, 10, 100)
            expected_events = []
            
            for rr in rr_range:
                exp_risk = min(custom_baseline_risk * rr, 1.0)
                exp_events = exposed_n * exp_risk + unexposed_n * custom_baseline_risk
                expected_events.append(exp_events)
            
            # Create the curve
            curve_fig = go.Figure()
            
            curve_fig.add_trace(go.Scatter(
                x=rr_range,
                y=expected_events,
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            
            # Add marker for current RR
            curve_fig.add_trace(go.Scatter(
                x=[custom_rr],
                y=[total_events],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Current RR'
            ))
            
            # Update layout
            curve_fig.update_layout(
                title="Expected Events by Relative Risk",
                xaxis_title="Relative Risk",
                yaxis_title="Expected Events",
                height=400,
                xaxis=dict(range=[1, 10])
            )
            
            st.plotly_chart(curve_fig, use_container_width=True)
        
        # Design comparison tab
        with tab_compare:
            st.subheader("Study Design Comparison: Cohort vs. Other Designs")
            
            # Create a 3-column layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("### Cohort Study")
                st.write("""
                **Approach**: Follow groups with and without exposure over time to see who develops disease
                
                **Strengths**:
                - Establishes temporality
                - Can study multiple outcomes
                - Directly measures incidence
                - Good for rare exposures
                
                **Limitations**:
                - Time and resource intensive
                - Loss to follow-up
                - Inefficient for rare diseases
                """)
            
            with col2:
                st.write("### Case-Control Study")
                st.write("""
                **Approach**: Compare past exposure in people with and without disease
                
                **Strengths**:
                - Efficient for rare diseases
                - Requires fewer subjects
                - Faster and less expensive
                - Can study multiple exposures
                
                **Limitations**:
                - Cannot directly compute incidence
                - Subject to recall bias
                - Selection of appropriate controls
                """)
            
            with col3:
                st.write("### Randomized Trial")
                st.write("""
                **Approach**: Randomly assign exposure and follow for outcomes
                
                **Strengths**:
                - Controls for confounding
                - Strongest for causality
                - Minimizes bias
                - Prospective design
                
                **Limitations**:
                - Ethical concerns
                - Expensive
                - Limited generalizability
                - Compliance issues
                """)
            
            # Add a decision flowchart
            st.subheader("When to Use Each Design")
            
            decision_fig = go.Figure()
            
            # Define decision points and positions
            decisions = [
                {"text": "Study Question", "x": 400, "y": 500},
                {"text": "Is the disease rare?", "x": 400, "y": 400},
                {"text": "Is intervention ethical?", "x": 250, "y": 300},
                {"text": "Are resources limited?", "x": 550, "y": 300},
                {"text": "Randomized Trial", "x": 250, "y": 200, "final": True},
                {"text": "Cohort Study", "x": 400, "y": 200, "final": True},
                {"text": "Case-Control Study", "x": 550, "y": 200, "final": True}
            ]
            
            # Draw boxes
            for decision in decisions:
                color = "green" if decision.get("final", False) else "blue"
                fill = "rgba(0, 128, 0, 0.2)" if decision.get("final", False) else "rgba(100, 149, 237, 0.3)"
                
                decision_fig.add_shape(
                    type="rect",
                    x0=decision["x"] - 80, x1=decision["x"] + 80,
                    y0=decision["y"] - 30, y1=decision["y"] + 30,
                    line=dict(color=color, width=2),
                    fillcolor=fill
                )
                
                decision_fig.add_annotation(
                    x=decision["x"], y=decision["y"],
                    text=decision["text"],
                    showarrow=False,
                    font=dict(size=12)
                )
            
            # Add connecting lines
            # Study question to first decision
            decision_fig.add_shape(
                type="line",
                x0=400, x1=400,
                y0=470, y1=430,
                line=dict(color="black", width=1)
            )
            
            # First decision to left and right
            decision_fig.add_shape(
                type="line",
                x0=400, x1=250,
                y0=370, y1=330,
                line=dict(color="black", width=1)
            )
            
            decision_fig.add_shape(
                type="line",
                x0=400, x1=550,
                y0=370, y1=330,
                line=dict(color="black", width=1)
            )
            
            # Left decision to RCT
            decision_fig.add_shape(
                type="line",
                x0=250, x1=250,
                y0=270, y1=230,
                line=dict(color="black", width=1)
            )
            
            # Right decision to Cohort and Case-Control
            decision_fig.add_shape(
                type="line",
                x0=550, x1=400,
                y0=270, y1=230,
                line=dict(color="black", width=1)
            )
            
            decision_fig.add_shape(
                type="line",
                x0=550, x1=550,
                y0=270, y1=230,
                line=dict(color="black", width=1)
            )
            
            # Add Yes/No labels
            decision_fig.add_annotation(
                x=320, y=385,
                text="No",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=480, y=385,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=250, y=250,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=470, y=285,
                text="No",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=550, y=250,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            # Update layout
            decision_fig.update_layout(
                height=550,
                plot_bgcolor="white",
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[150, 650]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[150, 550]
                )
            )
            
            st.plotly_chart(decision_fig, use_container_width=True)    
            
    # ENHANCED CASE-CONTROL STUDY SECTION
    elif study_design == "Case-Control Study":
        st.header("Case-Control Study Simulation")
        
        # Create tabs for different aspects of the visualization
        tab_main, tab_concepts, tab_interactive, tab_compare = st.tabs([
            "📉 Main Visualization", 
            "📝 Conceptual Overview", 
            "📊 Interactive Exploration", 
            "⚖️ Design Comparison"
        ])
        
        # Parameters for case-control study
        with tab_main:
            col1, col2 = st.columns(2)
            with col1:
                n_cases = st.slider("Number of Cases", 50, 500, 200, key="cc_cases")
                control_ratio = st.slider("Control to Case Ratio", 1, 4, 2, key="cc_ratio")
            with col2:
                odds_ratio = st.slider("True Odds Ratio", 1.0, 5.0, 2.0, key="cc_or")
                bg_exposure = st.slider("Background Exposure Prevalence", 0.1, 0.5, 0.3, key="cc_exposure")
        
        # Generate case-control data
        n_controls = n_cases * control_ratio
        
        # Calculate exposure probability for cases based on the true odds ratio
        case_exposure_prob = (odds_ratio * bg_exposure) / (1 - bg_exposure + odds_ratio * bg_exposure)
        
        # Generate data
        np.random.seed(42)
        
        # Cases
        case_exposure = np.random.binomial(1, case_exposure_prob, n_cases)
        case_data = pd.DataFrame({
            'Status': 'Case',
            'Exposed': case_exposure
        })
        
        # Controls
        control_exposure = np.random.binomial(1, bg_exposure, n_controls)
        control_data = pd.DataFrame({
            'Status': 'Control',
            'Exposed': control_exposure
        })
        
        # Combine datasets
        cc_data = pd.concat([case_data, control_data])
        
        # Create contingency table
        contingency_table = pd.crosstab(cc_data['Status'], cc_data['Exposed'], 
                                    margins=True, margins_name="Total")
        contingency_table.columns = ['Unexposed', 'Exposed', 'Total']
        
        # Calculate the observed odds ratio
        cases_exposed = contingency_table.loc['Case', 'Exposed']
        cases_unexposed = contingency_table.loc['Case', 'Unexposed']
        controls_exposed = contingency_table.loc['Control', 'Exposed']
        controls_unexposed = contingency_table.loc['Control', 'Unexposed']
        
        observed_or = (cases_exposed * controls_unexposed) / (cases_unexposed * controls_exposed)
        
        # Main visualization tab with step-by-step progression
        with tab_main:
            # Step-by-step progression slider
            steps = st.radio(
                "Study Design Stages:",
                ["1. Identify Cases", "2. Select Controls", "3. Assess Past Exposure", 
                "4. Compare Exposure Odds", "5. Interpret Results"],
                horizontal=True,
                key="case_control_steps"
            )
            
            # Create dynamic visualization based on selected step
            fig = go.Figure()
            
            # Constants for visualization
            total_width = 800
            total_height = 600
            timeline_y = 350
            
            # Draw timeline
            fig.add_shape(
                type="line",
                x0=50, x1=750,
                y0=timeline_y, y1=timeline_y,
                line=dict(color="black", width=3),
            )
            
            # Add "Past" and "Present" labels
            fig.add_annotation(
                x=150, y=timeline_y + 30,
                text="Past",
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.add_annotation(
                x=650, y=timeline_y + 30,
                text="Present",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Different elements based on step
            if steps == "1. Identify Cases":
                # Create case group at present
                fig.add_shape(
                    type="rect",
                    x0=600, x1=700,
                    y0=timeline_y - 120, y1=timeline_y - 20,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255, 0, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=650, y=timeline_y - 150,
                    text=f"Cases<br>n={n_cases}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw individual dots representing cases
                dot_rows = 5
                dot_cols = min(10, n_cases // dot_rows)
                
                for i in range(min(dot_rows * dot_cols, n_cases)):
                    row = i // dot_cols
                    col = i % dot_cols
                    
                    x_pos = 620 + col * 6
                    y_pos = timeline_y - 100 + row * 15
                    
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(color="red", size=6),
                        showlegend=False
                    ))
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=200, x1=550,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=375, y=timeline_y - 100,
                    text="Step 1: Identify Cases<br>" +
                        f"• Select {n_cases} individuals with the disease<br>" +
                        "• Cases are often identified from patient records<br>" +
                        "• Establish clear diagnostic criteria",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "2. Select Controls":
                # Draw cases
                fig.add_shape(
                    type="rect",
                    x0=600, x1=700,
                    y0=timeline_y - 120, y1=timeline_y - 20,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255, 0, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=650, y=timeline_y - 150,
                    text=f"Cases<br>n={n_cases}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw controls
                fig.add_shape(
                    type="rect",
                    x0=600, x1=700,
                    y0=timeline_y + 20, y1=timeline_y + 120,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0, 0, 255, 0.2)"
                )
                
                fig.add_annotation(
                    x=650, y=timeline_y + 150,
                    text=f"Controls<br>n={n_controls}<br>({control_ratio}:1 ratio)",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw individual dots representing cases and controls
                # Cases
                for i in range(min(20, n_cases)):
                    x_pos = 620 + (i % 5) * 10
                    y_pos = timeline_y - 100 + (i // 5) * 15
                    
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(color="red", size=6),
                        showlegend=False
                    ))
                
                # Controls
                for i in range(min(40, n_controls)):
                    x_pos = 620 + (i % 8) * 8
                    y_pos = timeline_y + 40 + (i // 8) * 12
                    
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(color="blue", size=6),
                        showlegend=False
                    ))
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=200, x1=550,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=375, y=timeline_y - 100,
                    text="Step 2: Select Controls<br>" +
                        f"• Select {n_controls} individuals without the disease<br>" +
                        f"• Control-to-case ratio: {control_ratio}:1<br>" +
                        "• Controls should represent the source population of cases",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
            elif steps == "3. Assess Past Exposure":
                # Draw cases and controls at present
                fig.add_shape(
                    type="rect",
                    x0=600, x1=700,
                    y0=timeline_y - 120, y1=timeline_y - 20,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255, 0, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=650, y=timeline_y - 70,
                    text=f"Cases<br>n={n_cases}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                fig.add_shape(
                    type="rect",
                    x0=600, x1=700,
                    y0=timeline_y + 20, y1=timeline_y + 120,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0, 0, 255, 0.2)"
                )
                
                fig.add_annotation(
                    x=650, y=timeline_y + 70,
                    text=f"Controls<br>n={n_controls}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw exposure assessment in the past
                # Exposure assessment for cases
                fig.add_shape(
                    type="rect",
                    x0=100, x1=200,
                    y0=timeline_y - 120, y1=timeline_y - 20,
                    line=dict(color="orange", width=2),
                    fillcolor="rgba(255, 165, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=150, y=timeline_y - 70,
                    text=f"Past Exposure<br>Cases: {cases_exposed}/{n_cases}<br>({cases_exposed/n_cases:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Exposure assessment for controls
                fig.add_shape(
                    type="rect",
                    x0=100, x1=200,
                    y0=timeline_y + 20, y1=timeline_y + 120,
                    line=dict(color="orange", width=2),
                    fillcolor="rgba(255, 165, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=150, y=timeline_y + 70,
                    text=f"Past Exposure<br>Controls: {controls_exposed}/{n_controls}<br>({controls_exposed/n_controls:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Use dotted lines to connect present to past instead of arrows
                fig.add_shape(
                    type="line",
                    x0=600, x1=200,
                    y0=timeline_y - 70, y1=timeline_y - 70,
                    line=dict(color="red", width=2, dash="dash"),
                )
                
                fig.add_shape(
                    type="line",
                    x0=600, x1=200,
                    y0=timeline_y + 70, y1=timeline_y + 70,
                    line=dict(color="blue", width=2, dash="dash"),
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=250, x1=550,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=400, y=timeline_y - 100,
                    text="Step 3: Assess Past Exposure<br>" +
                        "• Collect exposure history for both groups<br>" +
                        "• Methods include interviews, records, or specimens<br>" +
                        "• Goal is to determine if exposure differs between groups",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
            elif steps == "4. Compare Exposure Odds":
                # Create a 2x2 table visualization
                table_width = 400
                table_height = 200
                table_x = 400
                table_y = timeline_y
                
                # Draw table outline
                fig.add_shape(
                    type="rect",
                    x0=table_x - table_width/2, x1=table_x + table_width/2,
                    y0=table_y - table_height/2, y1=table_y + table_height/2,
                    line=dict(color="black", width=2),
                    fillcolor="white"
                )
                
                # Draw inner lines
                # Vertical line
                fig.add_shape(
                    type="line",
                    x0=table_x, x1=table_x,
                    y0=table_y - table_height/2, y1=table_y + table_height/2,
                    line=dict(color="black", width=1),
                )
                
                # Horizontal line
                fig.add_shape(
                    type="line",
                    x0=table_x - table_width/2, x1=table_x + table_width/2,
                    y0=table_y, y1=table_y,
                    line=dict(color="black", width=1),
                )
                
                # Add headers
                fig.add_annotation(
                    x=table_x - table_width/4, y=table_y + table_height/2 + 30,
                    text="Exposed",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                fig.add_annotation(
                    x=table_x + table_width/4, y=table_y + table_height/2 + 30,
                    text="Unexposed",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                fig.add_annotation(
                    x=table_x - table_width/2 - 50, y=table_y + table_height/4,
                    text="Cases",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                fig.add_annotation(
                    x=table_x - table_width/2 - 50, y=table_y - table_height/4,
                    text="Controls",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add cell values with cell shading
                # Cell a: Cases, Exposed
                fig.add_shape(
                    type="rect",
                    x0=table_x - table_width/2 + 1, x1=table_x - 1,
                    y0=table_y + 1, y1=table_y + table_height/2 - 1,
                    line=dict(color="black", width=1),
                    fillcolor="rgba(255, 0, 0, 0.15)"
                )
                
                fig.add_annotation(
                    x=table_x - table_width/4, y=table_y + table_height/4,
                    text=f"a = {cases_exposed}<br>({cases_exposed/n_cases:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Cell b: Cases, Unexposed
                fig.add_shape(
                    type="rect",
                    x0=table_x + 1, x1=table_x + table_width/2 - 1,
                    y0=table_y + 1, y1=table_y + table_height/2 - 1,
                    line=dict(color="black", width=1),
                    fillcolor="rgba(255, 0, 0, 0.15)"
                )
                
                fig.add_annotation(
                    x=table_x + table_width/4, y=table_y + table_height/4,
                    text=f"b = {cases_unexposed}<br>({cases_unexposed/n_cases:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Cell c: Controls, Exposed
                fig.add_shape(
                    type="rect",
                    x0=table_x - table_width/2 + 1, x1=table_x - 1,
                    y0=table_y - table_height/2 + 1, y1=table_y - 1,
                    line=dict(color="black", width=1),
                    fillcolor="rgba(0, 0, 255, 0.15)"
                )
                
                fig.add_annotation(
                    x=table_x - table_width/4, y=table_y - table_height/4,
                    text=f"c = {controls_exposed}<br>({controls_exposed/n_controls:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Cell d: Controls, Unexposed
                fig.add_shape(
                    type="rect",
                    x0=table_x + 1, x1=table_x + table_width/2 - 1,
                    y0=table_y - table_height/2 + 1, y1=table_y - 1,
                    line=dict(color="black", width=1),
                    fillcolor="rgba(0, 0, 255, 0.15)"
                )
                
                fig.add_annotation(
                    x=table_x + table_width/4, y=table_y - table_height/4,
                    text=f"d = {controls_unexposed}<br>({controls_unexposed/n_controls:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add formula for odds ratio
                fig.add_annotation(
                    x=table_x, y=table_y - table_height/2 - 50,
                    text=f"Odds Ratio (OR) = (a/b) ÷ (c/d) = ({cases_exposed}/{cases_unexposed}) ÷ ({controls_exposed}/{controls_unexposed}) = {observed_or:.2f}",
                    showarrow=False,
                    font=dict(size=14),
                    align="center",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=5,
                    bgcolor="white"
                )
                
                # Add explanation
                fig.add_annotation(
                    x=table_x, y=table_y + table_height/2 + 80,
                    text="Step 4: Compare Exposure Odds<br>" +
                        "• Calculate the odds of exposure in cases and controls<br>" +
                        "• Divide to get the odds ratio (OR)<br>" +
                        "• OR > 1 suggests exposure increases disease risk",
                    showarrow=False,
                    font=dict(size=14),
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=5,
                    bgcolor="white"
                )
                
            elif steps == "5. Interpret Results":
                # Create results visualization
                results_x = 400
                results_y = timeline_y
                
                # Create results box
                fig.add_shape(
                    type="rect",
                    x0=results_x - 150, x1=results_x + 150,
                    y0=results_y - 100, y1=results_y + 100,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.1)"
                )
                
                # Add OR result
                fig.add_annotation(
                    x=results_x, y=results_y + 50,
                    text=f"<b>Observed Odds Ratio: {observed_or:.2f}</b>",
                    showarrow=False,
                    font=dict(size=16),
                    align="center"
                )
                
                # Add interpretation text
                interpretation = ""
                if observed_or > 1.5:
                    interpretation = "Exposure appears to increase disease risk"
                elif observed_or < 0.67:
                    interpretation = "Exposure appears to decrease disease risk"
                else:
                    interpretation = "No strong association detected"
                    
                fig.add_annotation(
                    x=results_x, y=results_y,
                    text=f"Interpretation:<br>{interpretation}",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add estimation of true OR
                fig.add_annotation(
                    x=results_x, y=results_y - 50,
                    text=f"True OR specified: {odds_ratio:.2f}<br>Observed OR: {observed_or:.2f}",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=results_x - 250, x1=results_x + 250,
                    y0=results_y - 200, y1=results_y - 120,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=results_x, y=results_y - 160,
                    text="Step 5: Interpret Results<br>" +
                        "• OR = 1: No association between exposure and disease<br>" +
                        "• OR > 1: Exposure associated with increased disease odds<br>" +
                        "• OR < 1: Exposure associated with decreased disease odds",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add limitations box
                fig.add_shape(
                    type="rect",
                    x0=results_x - 250, x1=results_x + 250,
                    y0=results_y + 120, y1=results_y + 200,
                    line=dict(color="red", width=1),
                    fillcolor="rgba(255, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=results_x, y=results_y + 160,
                    text="<b>Key Limitations:</b><br>" +
                        "• Recall bias can affect exposure assessment<br>" +
                        "• Selection of appropriate controls is crucial<br>" +
                        "• Cannot directly estimate disease incidence",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
            
            # Update layout
            fig.update_layout(
                title="Case-Control Study Design",
                height=650,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[timeline_y - 250, timeline_y + 250]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Conceptual overview tab
        with tab_concepts:
            st.subheader("Case-Control Study: Conceptual Overview")
            
            st.write("""
            Case-control studies start with disease status and look backward to assess exposure. They're particularly useful for studying rare diseases.
            """)
            
            # Create a two-column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Key Strengths")
                st.write("""
                1. **Efficient for rare diseases**: Can study diseases with low prevalence
                2. **Resource efficient**: Requires fewer subjects than cohort studies
                3. **Multiple exposures**: Can assess many risk factors simultaneously
                4. **Quick results**: Faster than prospective designs
                """)
            
            with col2:
                st.write("#### Limitations")
                st.write("""
                1. **Recall bias**: Differential reporting of exposures
                2. **Selection bias**: Control group may not represent population
                3. **Temporality issues**: Cannot always establish sequence of events
                4. **No direct risk estimation**: Cannot directly calculate incidence
                """)
            
            # Create a simple process flow diagram
            process_fig = go.Figure()
            steps = ["Identify Cases", "Select Controls", "Assess Past Exposure", "Compare Odds", "Interpret Results"]
            x_positions = [100, 250, 400, 550, 700]
            y_position = 100
            
            # Draw process boxes
            for i, (step, x_pos) in enumerate(zip(steps, x_positions)):
                # Draw box
                process_fig.add_shape(
                    type="rect",
                    x0=x_pos-70, x1=x_pos+70,
                    y0=y_position-30, y1=y_position+30,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(100, 149, 237, 0.3)"
                )
                
                # Add step label
                process_fig.add_annotation(
                    x=x_pos, y=y_position,
                    text=f"{i+1}. {step}",
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                # Add connecting line (except for the last step)
                if i < len(steps) - 1:
                    next_x = x_positions[i+1]
                    process_fig.add_shape(
                        type="line",
                        x0=x_pos+70, x1=next_x-70,
                        y0=y_position, y1=y_position,
                        line=dict(color="blue", width=1),
                    )
            
            # Update layout
            process_fig.update_layout(
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[50, 150]
                )
            )
            
            st.plotly_chart(process_fig, use_container_width=True)
        
        # Interactive exploration tab
        with tab_interactive:
            st.subheader("Interactive Odds Ratio Explorer")
            
            # Let user adjust parameters
            col1, col2 = st.columns(2)
            
            with col1:
                custom_cases = st.slider(
                    "Number of Cases", 
                    50, 1000, 200, 
                    key="custom_cases"
                )
                custom_control_ratio = st.slider(
                    "Control-to-Case Ratio", 
                    1, 5, 2, 
                    key="custom_control_ratio"
                )
            
            with col2:
                custom_or = st.slider(
                    "True Odds Ratio", 
                    0.1, 10.0, 2.0, 
                    step=0.1,
                    key="custom_or"
                )
                custom_bg_exposure = st.slider(
                    "Background Exposure Prevalence", 
                    0.05, 0.5, 0.3, 
                    key="custom_bg_exposure"
                )
            
            # Calculate expected values
            custom_controls = custom_cases * custom_control_ratio
            
            # Calculate case exposure probability
            custom_case_exposure_prob = (custom_or * custom_bg_exposure) / (1 - custom_bg_exposure + custom_or * custom_bg_exposure)
            
            # Expected counts
            exp_cases_exposed = int(custom_cases * custom_case_exposure_prob)
            exp_cases_unexposed = custom_cases - exp_cases_exposed
            exp_controls_exposed = int(custom_controls * custom_bg_exposure)
            exp_controls_unexposed = custom_controls - exp_controls_exposed
            
            # Expected OR
            exp_or = (exp_cases_exposed * exp_controls_unexposed) / (exp_cases_unexposed * exp_controls_exposed)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cases", f"{custom_cases} total")
                st.metric("Exposed Cases", f"{exp_cases_exposed} ({exp_cases_exposed/custom_cases:.1%})")
                st.metric("Unexposed Cases", f"{exp_cases_unexposed} ({exp_cases_unexposed/custom_cases:.1%})")
            
            with col2:
                st.metric("Controls", f"{custom_controls} total")
                st.metric("Exposed Controls", f"{exp_controls_exposed} ({exp_controls_exposed/custom_controls:.1%})")
                st.metric("Unexposed Controls", f"{exp_controls_unexposed} ({exp_controls_unexposed/custom_controls:.1%})")
            
            with col3:
                st.metric("True Odds Ratio", f"{custom_or:.2f}")
                st.metric("Expected Observed OR", f"{exp_or:.2f}")
                
                # Calculate power
                # Simplified power calculation (not exact)
                n_total = custom_cases + custom_controls
                effect_size = abs(np.log(custom_or))
                power_estimate = 1 - 0.8 * np.exp(-0.0025 * n_total * effect_size)
                power_percent = min(power_estimate * 100, 99.9)
                
                st.metric("Estimated Power", f"{power_percent:.1f}%")
            
            # Create an interactive visualization of the 2x2 table
            st.subheader("Interactive 2×2 Table")
            
            # Create the visualization
            table_fig = go.Figure()
            
            # Create a 2x2 table
            table_width = 400
            table_height = 300
            
            # Draw table outline
            table_fig.add_shape(
                type="rect",
                x0=0, x1=table_width,
                y0=0, y1=table_height,
                line=dict(color="black", width=2),
                fillcolor="white"
            )
            
            # Draw inner lines
            # Vertical divider
            table_fig.add_shape(
                type="line",
                x0=table_width/2, x1=table_width/2,
                y0=0, y1=table_height,
                line=dict(color="black", width=1),
            )
            
            # Horizontal divider
            table_fig.add_shape(
                type="line",
                x0=0, x1=table_width,
                y0=table_height/2, y1=table_height/2,
                line=dict(color="black", width=1),
            )
            
            # Add headers
            table_fig.add_annotation(
                x=table_width/4, y=table_height + 30,
                text="Exposed",
                showarrow=False,
                font=dict(size=14)
            )
            
            table_fig.add_annotation(
                x=3*table_width/4, y=table_height + 30,
                text="Unexposed",
                showarrow=False,
                font=dict(size=14)
            )
            
            table_fig.add_annotation(
                x=-60, y=3*table_height/4,
                text="Cases",
                showarrow=False,
                font=dict(size=14)
            )
            
            table_fig.add_annotation(
                x=-60, y=table_height/4,
                text="Controls",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Add cell values with cell shading
            # Cell a: Cases, Exposed
            table_fig.add_shape(
                type="rect",
                x0=1, x1=table_width/2 - 1,
                y0=table_height/2 + 1, y1=table_height - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(255, 0, 0, 0.15)"
            )
            
            table_fig.add_annotation(
                x=table_width/4, y=3*table_height/4,
                text=f"a = {exp_cases_exposed}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Cell b: Cases, Unexposed
            table_fig.add_shape(
                type="rect",
                x0=table_width/2 + 1, x1=table_width - 1,
                y0=table_height/2 + 1, y1=table_height - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(255, 0, 0, 0.15)"
            )
            
            table_fig.add_annotation(
                x=3*table_width/4, y=3*table_height/4,
                text=f"b = {exp_cases_unexposed}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Cell c: Controls, Exposed
            table_fig.add_shape(
                type="rect",
                x0=1, x1=table_width/2 - 1,
                y0=1, y1=table_height/2 - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(0, 0, 255, 0.15)"
            )
            
            table_fig.add_annotation(
                x=table_width/4, y=table_height/4,
                text=f"c = {exp_controls_exposed}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Cell d: Controls, Unexposed
            table_fig.add_shape(
                type="rect",
                x0=table_width/2 + 1, x1=table_width - 1,
                y0=1, y1=table_height/2 - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(0, 0, 255, 0.15)"
            )
            
            table_fig.add_annotation(
                x=3*table_width/4, y=table_height/4,
                text=f"d = {exp_controls_unexposed}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Add OR calculation
            table_fig.add_annotation(
                x=table_width/2, y=-50,
                text=f"Odds Ratio (OR) = (a/b) ÷ (c/d) = ({exp_cases_exposed}/{exp_cases_unexposed}) ÷ ({exp_controls_exposed}/{exp_controls_unexposed}) = {exp_or:.2f}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Update layout
            table_fig.update_layout(
                title="Case-Control 2×2 Table",
                height=400,
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-100, table_width + 50]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-100, table_height + 50]
                )
            )
            
            st.plotly_chart(table_fig, use_container_width=True)
            
            # Add a visualization of OR interpretation
            st.subheader("Odds Ratio Interpretation")
            
            or_fig = go.Figure()
            
            # Create a scale visualization
            scale_width = 600
            center_x = scale_width / 2
            
            # Draw baseline (OR = 1)
            or_fig.add_shape(
                type="line",
                x0=0, x1=scale_width,
                y0=50, y1=50,
                line=dict(color="black", width=2),
            )
            
            # Draw tick marks
            ticks = [0.1, 0.25, 0.5, 1, 2, 4, 10]
            positions = [50, 125, 200, 300, 400, 475, 550]
            
            for tick, pos in zip(ticks, positions):
                # Draw tick
                or_fig.add_shape(
                    type="line",
                    x0=pos, x1=pos,
                    y0=45, y1=55,
                    line=dict(color="black", width=1),
                )
                
                # Add label
                or_fig.add_annotation(
                    x=pos, y=35,
                    text=str(tick),
                    showarrow=False,
                    font=dict(size=12)
                )
            
            # Add interpretation regions
            # Protective region
            or_fig.add_shape(
                type="rect",
                x0=0, x1=300,
                y0=60, y1=100,
                line=dict(color="green", width=1),
                fillcolor="rgba(0, 200, 0, 0.1)"
            )
            
            or_fig.add_annotation(
                x=150, y=80,
                text="Protective Association<br>OR < 1",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Harmful region
            or_fig.add_shape(
                type="rect",
                x0=300, x1=scale_width,
                y0=60, y1=100,
                line=dict(color="red", width=1),
                fillcolor="rgba(255, 0, 0, 0.1)"
            )
            
            or_fig.add_annotation(
                x=450, y=80,
                text="Harmful Association<br>OR > 1",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Add marker for the current OR
            or_position = 300  # Default (OR = 1)
            
            if custom_or < 1:
                # Scale for protective ORs
                or_position = 300 - (300 - 50) * (1 - custom_or) / 0.9
            else:
                # Scale for harmful ORs
                or_position = 300 + (scale_width - 300) * min(custom_or - 1, 9) / 9
            
            or_fig.add_shape(
                type="line",
                x0=or_position, x1=or_position,
                y0=0, y1=120,
                line=dict(color="blue", width=2, dash="dash"),
            )
            
            or_fig.add_annotation(
                x=or_position, y=120,
                text=f"Your OR: {custom_or:.2f}",
                showarrow=False,
                font=dict(size=14, color="blue"),
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1,
                borderpad=3
            )
            
            # Add neutral line
            or_fig.add_shape(
                type="line",
                x0=300, x1=300,
                y0=0, y1=120,
                line=dict(color="black", width=2, dash="dot"),
            )
            
            or_fig.add_annotation(
                x=300, y=10,
                text="No Association (OR = 1)",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Update layout
            or_fig.update_layout(
                title="Odds Ratio Interpretation Scale",
                height=500,
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-50, scale_width + 50]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 140]
                )
            )
            
            st.plotly_chart(or_fig, use_container_width=True)
        
        # Design comparison tab
        with tab_compare:
            st.subheader("Study Design Comparison: Case-Control vs. Other Designs")
            
            # Create a 3-column layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("### Case-Control Study")
                st.write("""
                **Approach**: Compare past exposure in people with and without disease
                
                **Strengths**:
                - Efficient for rare diseases
                - Requires fewer subjects
                - Can study multiple exposures
                - Relatively quick and inexpensive
                
                **Limitations**:
                - Recall bias
                - Selection bias
                - Cannot directly measure incidence
                - Temporal relationship may be unclear
                """)
            
            with col2:
                st.write("### Cohort Study")
                st.write("""
                **Approach**: Follow exposed and unexposed groups to see who develops disease
                
                **Strengths**:
                - Establishes temporality
                - Can study multiple outcomes
                - Directly measures incidence
                - Can calculate relative risks
                
                **Limitations**:
                - Inefficient for rare diseases
                - Loss to follow-up
                - Time and resource intensive
                - Selection bias at enrollment
                """)
            
            with col3:
                st.write("### Cross-sectional Study")
                st.write("""
                **Approach**: Assess exposure and disease simultaneously in a population
                
                **Strengths**:
                - Quick and inexpensive
                - No follow-up required
                - Good for prevalence estimation
                - Can study multiple exposures and outcomes
                
                **Limitations**:
                - Cannot establish temporality
                - Prevalence-incidence bias
                - Survival bias
                - Selection bias
                """)
            
            # Add a decision tree for study design selection
            st.subheader("When to Use a Case-Control Design")
            
            decision_tree = go.Figure()
            
            # Define decision points
            decisions = [
                {"text": "Is the disease rare?", "x": 400, "y": 300, "decision": True},
                {"text": "Are resources limited?", "x": 250, "y": 200, "decision": True},
                {"text": "Is temporal relationship clear?", "x": 550, "y": 200, "decision": True},
                {"text": "Case-Control Study", "x": 250, "y": 100, "decision": False, "final": True},
                {"text": "Nested Case-Control Study", "x": 400, "y": 100, "decision": False, "final": True},
                {"text": "Cohort Study", "x": 550, "y": 100, "decision": False, "final": True},
                {"text": "Cross-sectional Study", "x": 700, "y": 100, "decision": False, "final": True}
            ]
            
            # Draw decision nodes and endpoints
            for decision in decisions:
                box_color = "green" if decision.get("final", False) else "blue"
                box_fill = "rgba(0, 200, 0, 0.1)" if decision.get("final", False) else "rgba(200, 200, 255, 0.1)"
                
                # Draw box
                decision_tree.add_shape(
                    type="rect",
                    x0=decision["x"] - 75, x1=decision["x"] + 75,
                    y0=decision["y"] - 25, y1=decision["y"] + 25,
                    line=dict(color=box_color, width=2),
                    fillcolor=box_fill
                )
                
                # Add text
                decision_tree.add_annotation(
                    x=decision["x"], y=decision["y"],
                    text=decision["text"],
                    showarrow=False,
                    font=dict(size=11)
                )
            
            # Connect nodes with lines
            # First level
            decision_tree.add_shape(
                type="line",
                x0=400, x1=250,
                y0=275, y1=225,
                line=dict(color="black", width=1),
            )
            
            decision_tree.add_shape(
                type="line",
                x0=400, x1=550,
                y0=275, y1=225,
                line=dict(color="black", width=1),
            )
            
            # Second level - left branch
            decision_tree.add_shape(
                type="line",
                x0=250, x1=250,
                y0=175, y1=125,
                line=dict(color="black", width=1),
            )
            
            # Second level - right branch
            decision_tree.add_shape(
                type="line",
                x0=550, x1=400,
                y0=175, y1=125,
                line=dict(color="black", width=1),
            )
            
            decision_tree.add_shape(
                type="line",
                x0=550, x1=550,
                y0=175, y1=125,
                line=dict(color="black", width=1),
            )
            
            decision_tree.add_shape(
                type="line",
                x0=550, x1=700,
                y0=175, y1=125,
                line=dict(color="black", width=1),
            )
            
            # Add Yes/No labels
            decision_tree.add_annotation(
                x=310, y=250,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_tree.add_annotation(
                x=490, y=250,
                text="No",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_tree.add_annotation(
                x=250, y=150,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_tree.add_annotation(
                x=470, y=150,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_tree.add_annotation(
                x=550, y=150,
                text="No",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_tree.add_annotation(
                x=630, y=150,
                text="Unclear",
                showarrow=False,
                font=dict(size=10)
            )
            
            # Update layout
            decision_tree.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[150, 780]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[50, 350]
                )
            )
            
            st.plotly_chart(decision_tree, use_container_width=True)
                
    # ENHANCED RANDOMIZED CONTROLLED TRIAL SECTION
    elif study_design == "Randomized Controlled Trial":
        st.header("Randomized Controlled Trial Simulation")
        
        # Create tabs for different aspects of the visualization
        tab_main, tab_concepts, tab_interactive, tab_compare = st.tabs([
            "📉 Main Visualization", 
            "📝 Conceptual Overview", 
            "📊 Interactive Exploration", 
            "⚖️ Design Comparison"
        ])
        
        # Parameters for RCT (shared across all tabs)
        with tab_main:
            col1, col2 = st.columns(2)
            with col1:
                n_participants = st.slider("Number of Participants", 100, 1000, 400, key="rct_size")
                effect_size = st.slider("Treatment Effect Size", 0.0, 1.0, 0.3, key="rct_effect")
            with col2:
                binary_outcome = st.checkbox("Binary Outcome", value=True, 
                                        help="Simulate a binary (yes/no) outcome like recovery or a continuous outcome like blood pressure",
                                        key="rct_binary")
                allocation_ratio = st.slider("Treatment:Control Allocation Ratio", 1.0, 3.0, 1.0, step=0.5, key="rct_ratio",
                                        help="Ratio of participants assigned to treatment vs. control")
        
        # Generate RCT data
        np.random.seed(42)
        
        # Calculate group sizes based on allocation ratio
        total = n_participants
        n_treated = int(total * (allocation_ratio / (allocation_ratio + 1)))
        n_control = total - n_treated
        
        # Assign treatment with appropriate allocation ratio
        treatment_prob = n_treated / total
        treatment = np.random.binomial(1, treatment_prob, n_participants)
        
        if binary_outcome:
            # Binary outcome simulation
            control_event_rate = 0.3  # baseline event rate
            treatment_effect_odds = np.exp(-effect_size * 2)  # convert effect size to odds ratio
            treatment_event_rate = control_event_rate * treatment_effect_odds / (1 - control_event_rate + control_event_rate * treatment_effect_odds)
            
            # Generate outcomes
            outcome = np.zeros(n_participants)
            # Count the actual number of control and treatment subjects
            actual_n_control = sum(treatment == 0)
            actual_n_treatment = sum(treatment == 1)

            # Generate outcomes using the actual counts
            outcome[treatment == 0] = np.random.binomial(1, control_event_rate, actual_n_control)
            outcome[treatment == 1] = np.random.binomial(1, treatment_event_rate, actual_n_treatment)
                        
            # Create DataFrame
            rct_data = pd.DataFrame({
                'Patient_ID': range(1, n_participants + 1),
                'Treatment': ['Treatment' if t else 'Control' for t in treatment],
                'Outcome': ['Event' if o else 'No Event' for o in outcome]
            })
            
            # Calculate key statistics
            control_events = sum(outcome[treatment == 0])
            treated_events = sum(outcome[treatment == 1])
            
            control_rate = control_events / n_control
            treated_rate = treated_events / n_treated
            
            absolute_risk_reduction = control_rate - treated_rate
            relative_risk = treated_rate / control_rate if control_rate > 0 else 0
            number_needed_to_treat = 1 / absolute_risk_reduction if absolute_risk_reduction > 0 else float('inf')
            
        else:
            # Continuous outcome simulation
            baseline_mean = 100  # e.g., blood pressure baseline
            treatment_mean = baseline_mean - (effect_size * 15)  # treatment effect
            
            # Generate outcomes with some random variation
            outcome = np.zeros(n_participants)
            # Count the actual number of control and treatment subjects
            actual_n_control = sum(treatment == 0)
            actual_n_treatment = sum(treatment == 1)

            # Generate outcomes using the actual counts
            outcome[treatment == 0] = np.random.normal(baseline_mean, 15, actual_n_control)  # control group
            outcome[treatment == 1] = np.random.normal(treatment_mean, 15, actual_n_treatment)  # treatment group
            
            # Create DataFrame
            rct_data = pd.DataFrame({
                'Patient_ID': range(1, n_participants + 1),
                'Treatment': ['Treatment' if t else 'Control' for t in treatment],
                'Outcome': outcome
            })
            
            # Calculate key statistics
            control_mean = np.mean(outcome[treatment == 0])
            treated_mean = np.mean(outcome[treatment == 1])
            mean_difference = control_mean - treated_mean
        
        # Main visualization tab with step-by-step progression
        with tab_main:
            # Step-by-step progression slider
            steps = st.radio(
                "Study Design Stages:",
                ["1. Population Selection", "2. Randomization", "3. Intervention", 
                "4. Follow-up & Outcome Assessment", "5. Analysis & Interpretation"],
                horizontal=True,
                key="rct_steps"
            )
            
            # Create dynamic visualization based on selected step
            fig = go.Figure()
            
            # Constants for visualization
            total_width = 800
            total_height = 600
            center_x = total_width / 2
            timeline_y = 350
            
            # Different elements based on step
            if steps == "1. Population Selection":
                # Draw population box
                fig.add_shape(
                    type="rect",
                    x0=center_x - 150, x1=center_x + 150,
                    y0=timeline_y + 50, y1=timeline_y + 200,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y + 125,
                    text=f"Study Population<br>n={n_participants}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add inclusion/exclusion criteria
                criteria_text = (
                    "<b>Eligibility Criteria</b><br>" +
                    "• Inclusion criteria define who can participate<br>" +
                    "• Exclusion criteria define who cannot participate<br>" +
                    "• Criteria should be specific and clearly defined"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y - 50,
                    text=criteria_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=5,
                    bgcolor="white"
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=center_x - 250, x1=center_x + 250,
                    y0=timeline_y - 220, y1=timeline_y - 130,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y - 170,
                    text="Step 1: Define and recruit study population<br>" +
                        f"• Target sample size: {n_participants} participants<br>" +
                        "• Clear inclusion/exclusion criteria are essential",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
                # Add check for adequate sample size
                sample_size_text = "✅ Sample size appears adequate"
                sample_size_color = "green"
                
                if n_participants < 200:
                    sample_size_text = "⚠️ Sample size may be too small"
                    sample_size_color = "orange"
                elif n_participants > 800:
                    sample_size_text = "✅ Large sample size (high power)"
                    sample_size_color = "green"
                    
                fig.add_annotation(
                    x=center_x, y=timeline_y + 250,
                    text=sample_size_text,
                    showarrow=False,
                    font=dict(size=14, color=sample_size_color),
                    align="center"
                )
                
            elif steps == "2. Randomization":
                # Draw initial population
                fig.add_shape(
                    type="rect",
                    x0=center_x - 150, x1=center_x + 150,
                    y0=timeline_y + 150, y1=timeline_y + 250,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y + 200,
                    text=f"Study Population<br>n={n_participants}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw randomization box
                fig.add_shape(
                    type="rect",
                    x0=center_x - 150, x1=center_x + 150,
                    y0=timeline_y + 50, y1=timeline_y + 100,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(0, 0, 255, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y + 75,
                    text="Randomization Process",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Connect population to randomization
                fig.add_shape(
                    type="line",
                    x0=center_x, x1=center_x,
                    y0=timeline_y + 150, y1=timeline_y + 100,
                    line=dict(color="black", width=2),
                )
                
                # Draw treatment and control groups
                # Treatment group
                fig.add_shape(
                    type="rect",
                    x0=center_x - 200, x1=center_x - 50,
                    y0=timeline_y - 50, y1=timeline_y + 25,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 200, 0, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x - 125, y=timeline_y - 12.5,
                    text=f"Treatment Group<br>n={n_treated}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Control group
                fig.add_shape(
                    type="rect",
                    x0=center_x + 50, x1=center_x + 200,
                    y0=timeline_y - 50, y1=timeline_y + 25,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255, 0, 0, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x + 125, y=timeline_y - 12.5,
                    text=f"Control Group<br>n={n_control}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Connect randomization to groups
                fig.add_shape(
                    type="line",
                    x0=center_x - 50, x1=center_x - 125,
                    y0=timeline_y + 50, y1=timeline_y + 25,
                    line=dict(color="black", width=2),
                )
                
                fig.add_shape(
                    type="line",
                    x0=center_x + 50, x1=center_x + 125,
                    y0=timeline_y + 50, y1=timeline_y + 25,
                    line=dict(color="black", width=2),
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=center_x - 250, x1=center_x + 250,
                    y0=timeline_y - 170, y1=timeline_y - 55,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y - 112.5,
                    text="Step 2: Randomize participants to treatment groups<br>" +
                        f"• Treatment group: {n_treated} participants<br>" +
                        f"• Control group: {n_control} participants<br>" +
                        f"• Allocation ratio: {allocation_ratio}:1",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
                # Show randomization benefits
                benefits_text = (
                    "<b>Benefits of Randomization</b><br>" +
                    "• Balances known and unknown confounders<br>" +
                    "• Reduces selection bias<br>" +
                    "• Enables causal inference"
                )
                
                fig.add_annotation(
                    x=center_x - 300, y=timeline_y+50,
                    text=benefits_text,
                    showarrow=False,
                    font=dict(size=12),
                    align="left",
                    bordercolor="blue",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="white"
                )
                
            elif steps == "3. Intervention":
                # Draw treatment and control groups
                # Treatment group
                fig.add_shape(
                    type="rect",
                    x0=center_x - 200, x1=center_x - 50,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 200, 0, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x - 125, y=timeline_y + 100,
                    text=f"Treatment Group<br>n={n_treated}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Control group
                fig.add_shape(
                    type="rect",
                    x0=center_x + 50, x1=center_x + 200,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255, 0, 0, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x + 125, y=timeline_y + 100,
                    text=f"Control Group<br>n={n_control}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw intervention boxes
                # Treatment intervention
                fig.add_shape(
                    type="rect",
                    x0=center_x - 200, x1=center_x - 50,
                    y0=timeline_y - 50, y1=timeline_y + 25,
                    line=dict(color="purple", width=2),
                    fillcolor="rgba(128, 0, 128, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x - 125, y=timeline_y - 12.5,
                    text="Active Intervention",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Control intervention
                fig.add_shape(
                    type="rect",
                    x0=center_x + 50, x1=center_x + 200,
                    y0=timeline_y - 50, y1=timeline_y + 25,
                    line=dict(color="purple", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=center_x + 125, y=timeline_y - 12.5,
                    text="Placebo / Standard Care",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Connect groups to interventions
                fig.add_shape(
                    type="line",
                    x0=center_x - 125, x1=center_x - 125,
                    y0=timeline_y + 50, y1=timeline_y + 25,
                    line=dict(color="black", width=2),
                )
                
                fig.add_shape(
                    type="line",
                    x0=center_x + 125, x1=center_x + 125,
                    y0=timeline_y + 50, y1=timeline_y + 25,
                    line=dict(color="black", width=2),
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=center_x - 250, x1=center_x + 250,
                    y0=timeline_y - 170, y1=timeline_y - 55,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                # Explanation text varies by outcome type
                intervention_text = (
                    "Step 3: Administer intervention<br>" +
                    "• Treatment group receives active intervention<br>" +
                    "• Control group receives placebo or standard care<br>" +
                    f"• Expected effect size: {effect_size:.1f}"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y - 115,
                    text=intervention_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
                # Add blinding information
                blinding_text = (
                    "<b>Blinding Options</b><br>" +
                    "• Single-blind: Participants don't know their group<br>" +
                    "• Double-blind: Participants and researchers don't know<br>" +
                    "• Blinding reduces bias in outcome assessment"
                )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y - 240,
                    text=blinding_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=5,
                    bgcolor="white"
                )
                
            elif steps == "4. Follow-up & Outcome Assessment":
                # Draw timeline
                fig.add_shape(
                    type="line",
                    x0=center_x - 200, x1=center_x + 200,
                    y0=timeline_y, y1=timeline_y,
                    line=dict(color="black", width=2),
                )
                
                # Add timeline markers
                fig.add_annotation(
                    x=center_x - 200, y=timeline_y - 20,
                    text="Start",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                fig.add_annotation(
                    x=center_x + 200, y=timeline_y - 20,
                    text="End",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Draw treatment group follow-up
                fig.add_shape(
                    type="rect",
                    x0=center_x - 250, x1=center_x - 150,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 200, 0, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x - 200, y=timeline_y + 100,
                    text=f"Treatment Group<br>n={n_treated}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw control group follow-up
                fig.add_shape(
                    type="rect",
                    x0=center_x + 150, x1=center_x + 250,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255, 0, 0, 0.1)"
                )
                
                fig.add_annotation(
                    x=center_x + 200, y=timeline_y + 100,
                    text=f"Control Group<br>n={n_control}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw outcome assessment boxes
                if binary_outcome:
                    # Treatment outcomes
                    treatment_outcome_box_y = timeline_y - 100
                    
                    # "No Event" box for treatment
                    fig.add_shape(
                        type="rect",
                        x0=center_x - 250, x1=center_x - 200,
                        y0=treatment_outcome_box_y - 40, y1=treatment_outcome_box_y + 40,
                        line=dict(color="gray", width=2),
                        fillcolor="rgba(200, 200, 200, 0.3)"
                    )
                    
                    fig.add_annotation(
                        x=center_x - 225, y=treatment_outcome_box_y,
                        text=f"No Event<br>{n_treated - treated_events}<br>({(1-treated_rate):.1%})",
                        showarrow=False,
                        font=dict(size=12)
                    )
                    
                    # "Event" box for treatment
                    fig.add_shape(
                        type="rect",
                        x0=center_x - 200, x1=center_x - 150,
                        y0=treatment_outcome_box_y - 40, y1=treatment_outcome_box_y + 40,
                        line=dict(color="orange", width=2),
                        fillcolor="rgba(255, 165, 0, 0.2)"
                    )
                    
                    fig.add_annotation(
                        x=center_x - 175, y=treatment_outcome_box_y,
                        text=f"Event<br>{treated_events}<br>({treated_rate:.1%})",
                        showarrow=False,
                        font=dict(size=12)
                    )
                    
                    # Connect treatment group to outcomes
                    fig.add_shape(
                        type="line",
                        x0=center_x - 200, x1=center_x - 200,
                        y0=timeline_y + 50, y1=treatment_outcome_box_y + 40,
                        line=dict(color="black", width=1, dash="dash"),
                    )
                    
                    # Control outcomes
                    control_outcome_box_y = timeline_y - 100
                    
                    # "No Event" box for control
                    fig.add_shape(
                        type="rect",
                        x0=center_x + 150, x1=center_x + 200,
                        y0=control_outcome_box_y - 40, y1=control_outcome_box_y + 40,
                        line=dict(color="gray", width=2),
                        fillcolor="rgba(200, 200, 200, 0.3)"
                    )
                    
                    fig.add_annotation(
                        x=center_x + 175, y=control_outcome_box_y,
                        text=f"No Event<br>{n_control - control_events}<br>({(1-control_rate):.1%})",
                        showarrow=False,
                        font=dict(size=12)
                    )
                    
                    # "Event" box for control
                    fig.add_shape(
                        type="rect",
                        x0=center_x + 200, x1=center_x + 250,
                        y0=control_outcome_box_y - 40, y1=control_outcome_box_y + 40,
                        line=dict(color="orange", width=2),
                        fillcolor="rgba(255, 165, 0, 0.2)"
                    )
                    
                    fig.add_annotation(
                        x=center_x + 225, y=control_outcome_box_y,
                        text=f"Event<br>{control_events}<br>({control_rate:.1%})",
                        showarrow=False,
                        font=dict(size=12)
                    )
                    
                    # Connect control group to outcomes
                    fig.add_shape(
                        type="line",
                        x0=center_x + 200, x1=center_x + 200,
                        y0=timeline_y + 50, y1=control_outcome_box_y + 40,
                        line=dict(color="black", width=1, dash="dash"),
                    )
                else:
                    # Continuous outcome - show distribution
                    # Treatment outcome
                    treatment_outcome_box_y = timeline_y - 100
                    
                    fig.add_shape(
                        type="rect",
                        x0=center_x - 250, x1=center_x - 150,
                        y0=treatment_outcome_box_y - 40, y1=treatment_outcome_box_y + 40,
                        line=dict(color="green", width=2),
                        fillcolor="rgba(0, 200, 0, 0.1)"
                    )
                    
                    fig.add_annotation(
                        x=center_x - 200, y=treatment_outcome_box_y,
                        text=f"Mean Outcome<br>{treated_mean:.1f}",
                        showarrow=False,
                        font=dict(size=12)
                    )
                    
                    # Connect treatment group to outcome
                    fig.add_shape(
                        type="line",
                        x0=center_x - 200, x1=center_x - 200,
                        y0=timeline_y + 50, y1=treatment_outcome_box_y + 40,
                        line=dict(color="black", width=1, dash="dash"),
                    )
                    
                    # Control outcome
                    control_outcome_box_y = timeline_y - 100
                    
                    fig.add_shape(
                        type="rect",
                        x0=center_x + 150, x1=center_x + 250,
                        y0=control_outcome_box_y - 40, y1=control_outcome_box_y + 40,
                        line=dict(color="red", width=2),
                        fillcolor="rgba(255, 0, 0, 0.1)"
                    )
                    
                    fig.add_annotation(
                        x=center_x + 200, y=control_outcome_box_y,
                        text=f"Mean Outcome<br>{control_mean:.1f}",
                        showarrow=False,
                        font=dict(size=12)
                    )
                    
                    # Connect control group to outcome
                    fig.add_shape(
                        type="line",
                        x0=center_x + 200, x1=center_x + 200,
                        y0=timeline_y + 50, y1=control_outcome_box_y + 40,
                        line=dict(color="black", width=1, dash="dash"),
                    )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=center_x - 250, x1=center_x + 250,
                    y0=timeline_y - 280, y1=timeline_y - 160,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                if binary_outcome:
                    explanation_text = (
                        "Step 4: Follow-up and assess outcomes<br>" +
                        f"• Event rate in treatment group: {treated_rate:.1%}<br>" +
                        f"• Event rate in control group: {control_rate:.1%}<br>" + 
                        "• All participants are followed for the same duration"
                    )
                else:
                    explanation_text = (
                        "Step 4: Follow-up and assess outcomes<br>" +
                        f"• Mean outcome in treatment group: {treated_mean:.1f}<br>" +
                        f"• Mean outcome in control group: {control_mean:.1f}<br>" + 
                        "• All participants are followed for the same duration"
                    )
                
                fig.add_annotation(
                    x=center_x, y=timeline_y - 220,
                    text=explanation_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "5. Analysis & Interpretation":
                # Create results box
                results_box_y = timeline_y
                
                fig.add_shape(
                    type="rect",
                    x0=center_x - 200, x1=center_x + 200,
                    y0=results_box_y - 120, y1=results_box_y + 120,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.1)"
                )
                
                # Add results title
                fig.add_annotation(
                    x=center_x, y=results_box_y + 90,
                    text="<b>RCT Results</b>",
                    showarrow=False,
                    font=dict(size=16)
                )
                
                # Add result metrics based on outcome type
                if binary_outcome:
                    result_metrics = (
                        f"Treatment group: {treated_events}/{n_treated} ({treated_rate:.1%})<br>" +
                        f"Control group: {control_events}/{n_control} ({control_rate:.1%})<br><br>" +
                        f"Risk Ratio (RR): {relative_risk:.2f}<br>" +
                        f"Absolute Risk Reduction (ARR): {absolute_risk_reduction:.3f}<br>" +
                        f"Number Needed to Treat (NNT): {int(number_needed_to_treat) if number_needed_to_treat != float('inf') else 'N/A'}"
                    )
                else:
                    result_metrics = (
                        f"Treatment group mean: {treated_mean:.1f}<br>" +
                        f"Control group mean: {control_mean:.1f}<br><br>" +
                        f"Mean Difference: {mean_difference:.1f}<br>" +
                        f"Relative Reduction: {(mean_difference/control_mean):.1%}<br>" +
                        f"Effect Size (Cohen's d): {effect_size:.2f}"
                    )
                
                fig.add_annotation(
                    x=center_x, y=results_box_y,
                    text=result_metrics,
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add interpretation text
                if binary_outcome:
                    if relative_risk < 0.8:
                        interpretation = "Treatment REDUCED the event rate"
                        color = "green"
                    elif relative_risk > 1.2:
                        interpretation = "Treatment INCREASED the event rate"
                        color = "red"
                    else:
                        interpretation = "No significant effect detected"
                        color = "gray"
                else:
                    if mean_difference > 10:
                        interpretation = "Treatment IMPROVED the outcome"
                        color = "green"
                    elif mean_difference < -10:
                        interpretation = "Treatment WORSENED the outcome"
                        color = "red"
                    else:
                        interpretation = "No significant effect detected"
                        color = "gray"
                
                fig.add_annotation(
                    x=center_x, y=results_box_y - 90,
                    text=f"Interpretation: {interpretation}",
                    showarrow=False,
                    font=dict(size=14, color=color),
                    align="center"
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=center_x - 250, x1=center_x + 250,
                    y0=results_box_y - 250, y1=results_box_y - 150,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=center_x, y=results_box_y - 200,
                    text="Step 5: Analyze and interpret results<br>" +
                        "• Compare outcomes between treatment and control groups<br>" +
                        "• Calculate effect sizes and confidence intervals<br>" +
                        "• Determine clinical and statistical significance",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add validity considerations
                validity_text = (
                    "<b>Key RCT Validity Considerations</b><br>" +
                    "• Internal Validity: Was the study conducted properly?<br>" +
                    "• External Validity: Can the results be generalized?<br>" +
                    "• Statistical Validity: Was the analysis appropriate?"
                )
                
                fig.add_annotation(
                    x=center_x, y=results_box_y + 200,
                    text=validity_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="center",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=5,
                    bgcolor="white"
                )
            
            # Update layout
            fig.update_layout(
                title="Randomized Controlled Trial Design",
                height=total_height,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[timeline_y - 300, timeline_y + 300]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Optional animation for treatment vs control over time
            if st.checkbox("Show animated treatment effect", key="rct_animation"):
                progress_bar = st.progress(0)
                time_point = st.empty()
                animation_plot = st.empty()
                animation_description = st.empty()  # Add this line for the description
                
                # Create animation
                if binary_outcome:
                    # Binary outcome animation
                    frames = 50
                    
                    for i in range(frames + 1):
                        progress = int(100 * i / frames)
                        progress_bar.progress(progress)
                        
                        # Calculate cumulative event rates
                        # Simulate accumulating events over time
                        fraction_complete = i / frames
                        
                        cum_treated_events = int(treated_events * min(1, fraction_complete * 1.2))
                        cum_control_events = int(control_events * min(1, fraction_complete * 1.2))
                        
                        cum_treated_rate = cum_treated_events / n_treated if n_treated > 0 else 0
                        cum_control_rate = cum_control_events / n_control if n_control > 0 else 0
                        
                        # Create frame
                        anim_fig = go.Figure()
                        
                        # Add bars for treatment and control
                        anim_fig.add_trace(go.Bar(
                            x=['Treatment Group'],
                            y=[cum_treated_rate * 100],
                            name='Treatment',
                            marker_color='green'
                        ))
                        
                        anim_fig.add_trace(go.Bar(
                            x=['Control Group'],
                            y=[cum_control_rate * 100],
                            name='Control',
                            marker_color='red'
                        ))
                        
                        # Update layout
                        anim_fig.update_layout(
                            title=f"Cumulative Event Rate (Time Point: {fraction_complete:.1%})",
                            yaxis=dict(
                                title="Event Rate (%)",
                                range=[0, max(treated_rate, control_rate) * 100 * 1.2]
                            ),
                            height=400
                        )
                        
                        # Add annotation with current rates
                        anim_fig.add_annotation(
                            x=0.5, y=max(cum_treated_rate, cum_control_rate) * 100 * 1.1,
                            text=f"Treatment: {cum_treated_rate:.1%} vs. Control: {cum_control_rate:.1%}",
                            showarrow=False,
                            font=dict(size=14),
                            align="center"
                        )
                        
                        time_point.markdown(f"### Study Progress: {fraction_complete:.0%}")
                        animation_plot.plotly_chart(anim_fig, use_container_width=True)
                        
                        # Add description for binary outcome animation
                        with animation_description.container():
                            st.subheader("What's happening in this animation:")
                            
                            st.write(f"**Current study progress:** {fraction_complete:.0%} complete")
                            
                            st.write("**What you're seeing:**")
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.markdown("• <span style='color:green; font-weight:bold;'>Green bar:</span>", unsafe_allow_html=True)
                                st.markdown("• <span style='color:red; font-weight:bold;'>Red bar:</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.write(f"Treatment group event rate ({cum_treated_events} events out of {n_treated} participants)")
                                st.write(f"Control group event rate ({cum_control_events} events out of {n_control} participants)")
                            
                            st.write(f"**Current comparison:** Treatment: {cum_treated_rate:.1%} vs. Control: {cum_control_rate:.1%}")
                            
                            relative_risk = cum_treated_rate / cum_control_rate if cum_control_rate > 0 else float('inf')
                            absolute_risk_diff = cum_treated_rate - cum_control_rate
                            
                            st.write(f"**Relative Risk (RR):** {relative_risk:.2f}")
                            st.write(f"**Absolute Risk Difference:** {absolute_risk_diff:.1%}")
                            
                            st.write("This animation demonstrates how event rates accumulate in randomized controlled trials, "
                                    "allowing us to compare outcomes between treatment and control groups. "
                                    "The difference in event rates helps quantify the treatment effect.")
                            
                            # Add context-specific message based on progress
                            if fraction_complete < 0.25:
                                st.info("**Early trial data:** Initial trends are emerging, but it's too early to draw firm conclusions.")
                            elif fraction_complete < 0.75:
                                st.info("**Mid-trial data:** Patterns becoming clearer. The treatment effect is beginning to stabilize.")
                            else:
                                st.info("**Late-trial data:** Near final results. The observed difference represents the treatment effect that will be reported.")
                        
                        time_module.sleep(0.3)
                else:
                    # Continuous outcome animation
                    frames = 50
                    
                    for i in range(frames + 1):
                        progress = int(100 * i / frames)
                        progress_bar.progress(progress)
                        
                        # Simulate accumulating effect over time
                        fraction_complete = i / frames
                        
                        current_treated_mean = baseline_mean - (fraction_complete * mean_difference)
                        
                        # Create frame
                        anim_fig = go.Figure()
                        
                        # Add traces for treatment and control
                        anim_fig.add_trace(go.Scatter(
                            x=[fraction_complete * 100],
                            y=[current_treated_mean],
                            mode='markers',
                            name='Treatment',
                            marker=dict(color='green', size=12)
                        ))
                        
                        anim_fig.add_trace(go.Scatter(
                            x=[fraction_complete * 100],
                            y=[baseline_mean],
                            mode='markers',
                            name='Control',
                            marker=dict(color='red', size=12)
                        ))
                        
                        # Add line traces for trends
                        anim_fig.add_trace(go.Scatter(
                            x=list(range(int(fraction_complete * 100) + 1)),
                            y=[baseline_mean - (x/100 * mean_difference) for x in range(int(fraction_complete * 100) + 1)],
                            mode='lines',
                            line=dict(color='green', width=2),
                            name='Treatment Trend'
                        ))
                        
                        anim_fig.add_trace(go.Scatter(
                            x=list(range(int(fraction_complete * 100) + 1)),
                            y=[baseline_mean] * (int(fraction_complete * 100) + 1),
                            mode='lines',
                            line=dict(color='red', width=2),
                            name='Control Trend'
                        ))
                        
                        # Update layout
                        anim_fig.update_layout(
                            title=f"Treatment Effect Over Time (Progress: {fraction_complete:.0%})",
                            xaxis=dict(
                                title="Study Progress (%)",
                                range=[0, 100]
                            ),
                            yaxis=dict(
                                title="Outcome Measure",
                                range=[min(treated_mean, control_mean) - 10, max(treated_mean, control_mean) + 10]
                            ),
                            height=400
                        )
                        
                        # Add annotation with current means
                        anim_fig.add_annotation(
                            x=50, y=max(baseline_mean, current_treated_mean) + 8,
                            text=f"Treatment: {current_treated_mean:.1f} vs. Control: {baseline_mean:.1f}",
                            showarrow=False,
                            font=dict(size=14),
                            align="center"
                        )
                        
                        time_point.markdown(f"### Study Progress: {fraction_complete:.0%}")
                        animation_plot.plotly_chart(anim_fig, use_container_width=True)
                        
                        # Add description for continuous outcome animation
                        with animation_description.container():
                            st.subheader("What's happening in this animation:")
                            
                            st.write(f"**Current study progress:** {fraction_complete:.0%} complete")
                            
                            st.write("**What you're seeing:**")
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.markdown("• <span style='color:green; font-weight:bold;'>Green line/point:</span>", unsafe_allow_html=True)
                                st.markdown("• <span style='color:red; font-weight:bold;'>Red line/point:</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.write("Treatment group outcome measure trend")
                                st.write("Control group outcome measure (baseline)")
                            
                            st.write(f"**Current comparison:** Treatment: {current_treated_mean:.1f} vs. Control: {baseline_mean:.1f}")
                            
                            # Calculate effect size (Cohen's d) as study progresses
                            if fraction_complete > 0:
                                current_diff = baseline_mean - current_treated_mean
                                pooled_sd = (treated_sd + control_sd) / 2
                                cohens_d = current_diff / pooled_sd if pooled_sd > 0 else 0
                                st.write(f"**Mean Difference:** {current_diff:.1f}")
                                st.write(f"**Effect Size (Cohen's d):** {cohens_d:.2f}")
                            
                            st.write("This animation demonstrates how treatment effects develop over the course of a randomized controlled trial, "
                                    "showing the gradual separation between treatment and control group means. "
                                    "In this case, lower values represent better outcomes.")
                            
                            # Add context-specific message based on progress
                            if fraction_complete < 0.25:
                                st.info("**Early trial data:** The treatment effect is beginning to emerge, but variability is still high.")
                            elif fraction_complete < 0.75:
                                st.info("**Mid-trial data:** The treatment effect is becoming more pronounced as more data accumulates.")
                            else:
                                st.info("**Late-trial data:** The final treatment effect is emerging, showing the true impact of the intervention.")
                        
                        time_module.sleep(0.3)
        # Conceptual overview tab
        with tab_concepts:
            st.subheader("Randomized Controlled Trial: Conceptual Overview")
            
            st.write("""
            Randomized Controlled Trials (RCTs) are considered the gold standard for studying causal relationships 
            between interventions and outcomes. By randomly assigning participants to treatment or control groups, 
            RCTs minimize bias and confounding.
            """)
            
            # Create a two-column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Key Strengths")
                st.write("""
                1. **Randomization**: Balances known and unknown confounders
                2. **Control group**: Provides comparison to assess intervention effects
                3. **Blinding**: Reduces observer and participant bias
                4. **Causal inference**: Strongest design for causality
                """)
            
            with col2:
                st.write("#### Limitations")
                st.write("""
                1. **Resource intensive**: Often expensive and time-consuming
                2. **Ethical constraints**: Not all interventions can be randomized
                3. **External validity**: May have limited generalizability
                4. **Non-compliance**: Participants may not adhere to assigned treatment
                """)
            
            # Create a simple process flow diagram
            process_fig = go.Figure()
            steps = ["Population Selection", "Randomization", "Intervention", "Follow-up", "Analysis"]
            x_positions = [100, 250, 400, 550, 700]
            y_position = 100
            
            # Draw process boxes
            for i, (step, x_pos) in enumerate(zip(steps, x_positions)):
                # Draw box
                process_fig.add_shape(
                    type="rect",
                    x0=x_pos-70, x1=x_pos+70,
                    y0=y_position-30, y1=y_position+30,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(100, 149, 237, 0.3)"
                )
                
                # Add step label
                process_fig.add_annotation(
                    x=x_pos, y=y_position,
                    text=f"{i+1}. {step}",
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                # Add connecting line (except for the last step)
                if i < len(steps) - 1:
                    next_x = x_positions[i+1]
                    process_fig.add_shape(
                        type="line",
                        x0=x_pos+70, x1=next_x-70,
                        y0=y_position, y1=y_position,
                        line=dict(color="blue", width=1),
                    )
            
            # Update layout
            process_fig.update_layout(
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[50, 150]
                )
            )
            
            st.plotly_chart(process_fig, use_container_width=True)
        
        # Interactive exploration tab
        with tab_interactive:
            st.subheader("Interactive Treatment Effect Explorer")
            
            # Let user adjust parameters
            col1, col2 = st.columns(2)
            
            with col1:
                custom_n = st.slider(
                    "Sample Size", 
                    50, 2000, 400, 
                    step=50,
                    key="custom_n"
                )
                
                if binary_outcome:
                    custom_baseline = st.slider(
                        "Baseline Event Rate (Control)", 
                        0.05, 0.5, 0.3, 
                        key="custom_baseline"
                    )
                else:
                    custom_baseline = st.slider(
                        "Baseline Mean (Control)", 
                        50, 150, 100, 
                        key="custom_baseline"
                    )
            
            with col2:
                custom_effect = st.slider(
                    "Effect Size", 
                    0.0, 1.0, 0.3, 
                    step=0.05,
                    key="custom_effect"
                )
                
                custom_allocation = st.slider(
                    "Treatment:Control Allocation", 
                    1.0, 3.0, 1.0, 
                    step=0.5,
                    key="custom_allocation"
                )
            
            # Calculate expected results based on user parameters
            n_treatment = int(custom_n * (custom_allocation / (custom_allocation + 1)))
            n_control = custom_n - n_treatment
            
            if binary_outcome:
                # Binary outcome calculations
                treatment_effect_odds = np.exp(-custom_effect * 2)
                treatment_rate = custom_baseline * treatment_effect_odds / (1 - custom_baseline + custom_baseline * treatment_effect_odds)
                
                treatment_events = int(n_treatment * treatment_rate)
                control_events = int(n_control * custom_baseline)
                
                treatment_rate = treatment_events / n_treatment
                control_rate = control_events / n_control
                
                abs_risk_reduction = control_rate - treatment_rate
                rel_risk = treatment_rate / control_rate if control_rate > 0 else 0
                nnt = 1 / abs_risk_reduction if abs_risk_reduction > 0 else float('inf')
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Treatment Group", f"{n_treatment} participants")
                    st.metric("Events in Treatment", f"{treatment_events} ({treatment_rate:.1%})")
                
                with col2:
                    st.metric("Control Group", f"{n_control} participants")
                    st.metric("Events in Control", f"{control_events} ({control_rate:.1%})")
                
                with col3:
                    st.metric("Relative Risk", f"{rel_risk:.2f}")
                    st.metric("Absolute Risk Reduction", f"{abs_risk_reduction:.3f}")
                    
                    if abs_risk_reduction > 0:
                        st.metric("Number Needed to Treat", f"{int(nnt)}")
            else:
                # Continuous outcome calculations
                treatment_mean = custom_baseline - (custom_effect * 15)
                mean_diff = custom_baseline - treatment_mean
                
                # Simulate standard deviation
                std_dev = 15
                
                # Calculate statistical significance (simplified)
                se = std_dev * np.sqrt(1/n_treatment + 1/n_control)
                t_stat = mean_diff / se
                sig = "Yes" if abs(t_stat) > 1.96 else "No"
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Treatment Group", f"{n_treatment} participants")
                    st.metric("Treatment Mean", f"{treatment_mean:.1f}")
                
                with col2:
                    st.metric("Control Group", f"{n_control} participants")
                    st.metric("Control Mean", f"{custom_baseline:.1f}")
                
                with col3:
                    st.metric("Mean Difference", f"{mean_diff:.1f}")
                    st.metric("Effect Size (Cohen's d)", f"{mean_diff/std_dev:.2f}")
                    st.metric("Statistically Significant", sig)
            
            # Create result visualization
            st.subheader("Visualized Treatment Effect")
            
            if binary_outcome:
                # Binary outcome visualization
                bin_fig = px.bar(
                    x=["Treatment", "Control"],
                    y=[treatment_rate * 100, control_rate * 100],
                    color=["Treatment", "Control"],
                    color_discrete_map={"Treatment": "green", "Control": "red"},
                    labels={"x": "Group", "y": "Event Rate (%)"}
                )
                
                # Update layout
                bin_fig.update_layout(
                    title="Comparison of Event Rates",
                    showlegend=False,
                    height=400,
                    yaxis=dict(range=[0, max(treatment_rate, control_rate) * 100 * 1.2])
                )
                
                # Add annotation for ARR
                bin_fig.add_annotation(
                    x=0.5, y=max(treatment_rate, control_rate) * 100 * 1.1,
                    text=f"Absolute Risk Reduction: {abs_risk_reduction:.3f} ({abs_risk_reduction/control_rate:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                st.plotly_chart(bin_fig, use_container_width=True)
                
                # Add NNT visualization
                if abs_risk_reduction > 0 and nnt < 100:
                    st.subheader(f"Number Needed to Treat (NNT): {int(nnt)}")
                    
                    # Create NNT visualization
                    dots_per_row = 10
                    total_dots = 100
                    rows = total_dots // dots_per_row
                    
                    nnt_fig = go.Figure()
                    
                    # Draw dots
                    for i in range(total_dots):
                        row = i // dots_per_row
                        col = i % dots_per_row
                        
                        x_pos = col * 20
                        y_pos = row * 20
                        
                        # Color dots based on NNT
                        color = "green" if i < nnt else "gray"
                        
                        nnt_fig.add_shape(
                            type="circle",
                            x0=x_pos - 7, x1=x_pos + 7,
                            y0=y_pos - 7, y1=y_pos + 7,
                            fillcolor=color,
                            line_color="black",
                            line_width=1
                        )
                    
                    # Add explanation
                    nnt_fig.add_annotation(
                        x=100, y=100,
                        text=f"Each green dot represents a patient who benefits from treatment<br>NNT = {int(nnt)} means you need to treat {int(nnt)} patients<br>for 1 patient to benefit",
                        showarrow=False,
                        font=dict(size=14),
                        align="center",
                        bgcolor="white",
                        bordercolor="black",
                        borderpad=5
                    )
                    
                    # Update layout
                    nnt_fig.update_layout(
                        height=300,
                        showlegend=False,
                        xaxis=dict(
                            showticklabels=False,
                            showgrid=False,
                            zeroline=False,
                            range=[-10, 210]
                        ),
                        yaxis=dict(
                            showticklabels=False,
                            showgrid=False,
                            zeroline=False,
                            range=[-10, 110]
                        )
                    )
                    
                    st.plotly_chart(nnt_fig, use_container_width=True)
            else:
                # Continuous outcome visualization
                # Create box plots for distributions
                # Generate sample data
                np.random.seed(42)
                treatment_samples = np.random.normal(treatment_mean, std_dev, n_treatment)
                control_samples = np.random.normal(custom_baseline, std_dev, n_control)
                
                # Create dataframe for plotting
                plot_data = pd.DataFrame({
                    'Group': ['Treatment'] * n_treatment + ['Control'] * n_control,
                    'Value': np.concatenate([treatment_samples, control_samples])
                })
                
                cont_fig = px.box(
                    plot_data,
                    x='Group',
                    y='Value',
                    color='Group',
                    color_discrete_map={'Treatment': 'green', 'Control': 'red'},
                    points="all"
                )
                
                # Add mean lines
                cont_fig.add_shape(
                    type="line",
                    x0=-0.4, x1=0.4,
                    y0=treatment_mean, y1=treatment_mean,
                    line=dict(color="darkgreen", width=3, dash="dash"),
                )
                
                cont_fig.add_shape(
                    type="line",
                    x0=0.6, x1=1.4,
                    y0=custom_baseline, y1=custom_baseline,
                    line=dict(color="darkred", width=3, dash="dash"),
                )
                
                # Update layout
                cont_fig.update_layout(
                    title="Distribution of Outcomes by Group",
                    showlegend=False,
                    height=500,
                    yaxis_title="Outcome Value"
                )
                
                # Add annotation for mean difference
                cont_fig.add_annotation(
                    x=0.5, y=max(treatment_mean, custom_baseline) + std_dev * 1.5,
                    text=f"Mean Difference: {mean_diff:.1f} ({mean_diff/custom_baseline:.1%})",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                st.plotly_chart(cont_fig, use_container_width=True)
            
            # Add power calculation
            st.subheader("Statistical Power Analysis")
            
            # Simple power calculation (not exact)
            if binary_outcome:
                effect = abs(treatment_rate - control_rate)
                pooled_p = (treatment_events + control_events) / (n_treatment + n_control)
                se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_treatment + 1/n_control))
                z = effect / se
                power = norm.cdf(z - 1.96) + norm.cdf(-z - 1.96)
            else:
                # For continuous outcomes
                cohen_d = mean_diff / std_dev
                se = np.sqrt(1/n_treatment + 1/n_control)
                ncp = cohen_d / se
                power = 1 - norm.cdf(1.96 - ncp) + norm.cdf(-1.96 - ncp)
            
            # Display power
            power_pct = min(power * 100, 99.9)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Statistical Power", f"{power_pct:.1f}%")
                
                if power_pct < 80:
                    st.warning("Power is below the conventional 80% threshold. Consider increasing sample size.")
                else:
                    st.success(f"Power is adequate (>80%) to detect the specified effect size.")
            
            with col2:
                # Create power meter visualization
                power_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=power_pct,
                    domain=dict(x=[0, 1], y=[0, 1]),
                    title=dict(text="Statistical Power"),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color="green"),
                        steps=[
                            dict(range=[0, 50], color="red"),
                            dict(range=[50, 80], color="orange"),
                            dict(range=[80, 100], color="lightgreen")
                        ],
                        threshold=dict(
                            line=dict(color="black", width=4),
                            thickness=0.75,
                            value=80
                        )
                    )
                ))
                
                power_fig.update_layout(height=250)
                st.plotly_chart(power_fig, use_container_width=True)
        
        # Design comparison tab
        with tab_compare:
            st.subheader("Study Design Comparison: RCT vs. Other Designs")
            
            # Create a 3-column layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("### Randomized Controlled Trial")
                st.write("""
                **Approach**: Randomly assign participants to intervention or control groups
                
                **Strengths**:
                - Gold standard for causality
                - Balances confounders through randomization
                - Minimizes selection and observer bias
                - Allows for blinding
                
                **Limitations**:
                - Resource intensive
                - Ethical constraints
                - Artificial conditions
                - Compliance issues
                """)
            
            with col2:
                st.write("### Cohort Study")
                st.write("""
                **Approach**: Follow exposed and unexposed groups over time
                
                **Strengths**:
                - Can study multiple outcomes
                - Establishes temporality
                - More practical for long-term exposures
                - Better represents real-world conditions
                
                **Limitations**:
                - Susceptible to confounding
                - Selection bias at baseline
                - Loss to follow-up
                - Less control over exposures
                """)
            
            with col3:
                st.write("### Case-Control Study")
                st.write("""
                **Approach**: Compare past exposures in cases vs. controls
                
                **Strengths**:
                - Efficient for rare outcomes
                - Requires fewer participants
                - Faster results
                - Can study multiple exposures
                
                **Limitations**:
                - Recall bias
                - Selection bias in control group
                - Cannot directly calculate risks
                - Temporal relationship issues
                """)
            
            # Add a comparison of evidence quality
            st.subheader("Hierarchy of Evidence")
            
            evidence_fig = go.Figure()
            
            # Create pyramid levels
            levels = [
                {"name": "Meta-analyses and\nSystematic Reviews", "height": 50, "width": 150, "y": 450, "color": "rgba(0, 100, 0, 0.7)"},
                {"name": "Randomized\nControlled Trials", "height": 75, "width": 250, "y": 375, "color": "rgba(50, 150, 50, 0.7)"},
                {"name": "Cohort Studies", "height": 75, "width": 350, "y": 300, "color": "rgba(100, 200, 100, 0.7)"},
                {"name": "Case-Control Studies", "height": 75, "width": 450, "y": 225, "color": "rgba(150, 250, 150, 0.7)"},
                {"name": "Cross-sectional Studies", "height": 75, "width": 550, "y": 150, "color": "rgba(200, 255, 200, 0.7)"},
                {"name": "Case Series and\nCase Reports", "height": 75, "width": 650, "y": 75, "color": "rgba(220, 255, 220, 0.7)"},
                {"name": "Expert Opinion", "height": 75, "width": 750, "y": 0, "color": "rgba(240, 255, 240, 0.7)"}
            ]
            
            # Draw pyramid levels
            for level in levels:
                half_width = level["width"] / 2
                
                # Draw pyramid level
                evidence_fig.add_shape(
                    type="rect",
                    x0=400 - half_width, x1=400 + half_width,
                    y0=level["y"], y1=level["y"] + level["height"],
                    line=dict(color="black", width=1),
                    fillcolor=level["color"]
                )
                
                # Add label
                evidence_fig.add_annotation(
                    x=400, y=level["y"] + level["height"]/2,
                    text=level["name"],
                    showarrow=False,
                    font=dict(size=12)
                )
            
            # Highlight RCTs
            evidence_fig.add_shape(
                type="rect",
                x0=400 - 125, x1=400 + 125,
                y0=375, y1=450,
                line=dict(color="blue", width=3),
                fillcolor="rgba(0, 0, 0, 0)",
            )
            
            # Add labels for the pyramid
            evidence_fig.add_annotation(
                x=400, y=550,
                text="Hierarchy of Evidence",
                showarrow=False,
                font=dict(size=16, color="black")
            )
            
            evidence_fig.add_annotation(
                x=100, y=250,
                text="Higher Quality Evidence",
                showarrow=False,
                font=dict(size=14),
                textangle=90
            )
            
            # Add legend
            evidence_fig.add_annotation(
                x=700, y=500,
                text="Key features of stronger evidence:<br>" +
                    "• Randomization<br>" +
                    "• Control groups<br>" +
                    "• Blinding<br>" +
                    "• Large sample size<br>" +
                    "• Representative population",
                showarrow=False,
                font=dict(size=12),
                align="left",
                bordercolor="black",
                borderwidth=1,
                borderpad=5,
                bgcolor="white"
            )
            
            # Update layout
            evidence_fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 600]
                )
            )
            
            st.plotly_chart(evidence_fig, use_container_width=True)
        
    # ENHANCED CROSS-SECTIONAL STUDY SECTION
    elif study_design == "Cross-sectional Study":
        st.header("Cross-sectional Study Simulation")
        
        # Create tabs for different aspects of the visualization
        tab_main, tab_concepts, tab_interactive, tab_compare = st.tabs([
            "📉 Main Visualization", 
            "📝 Conceptual Overview", 
            "📊 Interactive Exploration", 
            "⚖️ Design Comparison"
        ])
        
        # Parameters for cross-sectional study
        with tab_main:
            col1, col2 = st.columns(2)
            with col1:
                n_participants = st.slider("Number of Participants", 100, 2000, 500, key="cs_participants")
                prevalence = st.slider("Disease Prevalence", 0.05, 0.50, 0.20, key="cs_prevalence")
            with col2:
                exposure_prevalence = st.slider("Exposure Prevalence", 0.10, 0.70, 0.30, key="cs_exposure")
                association_strength = st.slider("Association Strength (PR)", 1.0, 5.0, 2.0, key="cs_strength")
        
        # Generate cross-sectional data
        def generate_cross_sectional_data(n, disease_prev, exp_prev, prevalence_ratio):
            # Generate exposure status
            exposure = np.random.binomial(1, exp_prev, n)
            
            # Calculate disease probability based on exposure
            # Using prevalence ratio to determine disease probability in exposed
            prob_unexposed = disease_prev / (exp_prev * prevalence_ratio + (1 - exp_prev))
            prob_exposed = prob_unexposed * prevalence_ratio
            
            # Generate disease status
            disease = np.zeros(n)
            disease[exposure == 0] = np.random.binomial(1, prob_unexposed, sum(exposure == 0))
            disease[exposure == 1] = np.random.binomial(1, prob_exposed, sum(exposure == 1))
            
            # Add some demographic variables
            age = np.random.normal(45, 15, n)
            gender = np.random.binomial(1, 0.5, n)
            
            return pd.DataFrame({
                'ID': range(1, n+1),
                'Exposure': exposure,
                'Disease': disease,
                'Age': age,
                'Gender': gender
            })

        # Generate data
        np.random.seed(42)
        cs_data = generate_cross_sectional_data(
            n_participants, 
            prevalence,
            exposure_prevalence,
            association_strength
        )

        # Calculate key statistics
        # Create 2x2 table
        contingency = pd.crosstab(cs_data['Exposure'], cs_data['Disease'], margins=True, margins_name="Total")
        contingency.columns = ['No Disease', 'Disease', 'Total']
        contingency.index = ['Unexposed', 'Exposed', 'Total']
        
        # Calculate counts
        exposed_with_disease = contingency.loc['Exposed', 'Disease']
        exposed_total = contingency.loc['Exposed', 'Total']
        unexposed_with_disease = contingency.loc['Unexposed', 'Disease']
        unexposed_total = contingency.loc['Unexposed', 'Total']
        total_with_disease = contingency.loc['Total', 'Disease']
        
        # Calculate prevalence
        prevalence_exposed = exposed_with_disease / exposed_total
        prevalence_unexposed = unexposed_with_disease / unexposed_total
        overall_prevalence = total_with_disease / n_participants
        
        # Calculate prevalence ratio and odds ratio
        prevalence_ratio = prevalence_exposed / prevalence_unexposed
        
        odds_exposed = exposed_with_disease / (exposed_total - exposed_with_disease)
        odds_unexposed = unexposed_with_disease / (unexposed_total - unexposed_with_disease)
        odds_ratio = odds_exposed / odds_unexposed
        
        # Main visualization tab with step-by-step progression
        with tab_main:
            # Step-by-step progression slider
            steps = st.radio(
                "Study Design Stages:",
                ["1. Population Selection", "2. Single Time Point Assessment", 
                "3. Measure Exposure & Disease", "4. Analysis & Interpretation"],
                horizontal=True,
                key="cs_steps"
            )
            
            # Create dynamic visualization based on selected step
            fig = go.Figure()
            
            # Constants for visualization
            total_width = 800
            total_height = 600
            center_x = total_width / 2
            center_y = total_height / 2
            
            if steps == "1. Population Selection":
                # Draw population selection
                fig.add_shape(
                    type="rect",
                    x0=center_x - 150, x1=center_x + 150,
                    y0=center_y - 100, y1=center_y + 100,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(100, 149, 237, 0.3)"
                )
                
                fig.add_annotation(
                    x=center_x, y=center_y,
                    text=f"Study Population<br>n={n_participants}",
                    showarrow=False,
                    font=dict(size=16)
                )
                

                
                # Add explanation
                explanation_x = center_x
                explanation_y = center_y - 200
                
                fig.add_shape(
                    type="rect",
                    x0=explanation_x - 250, x1=explanation_x + 250,
                    y0=explanation_y - 70, y1=explanation_y + 60,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=explanation_x, y=explanation_y,
                    text="Step 1: Define study population<br>" +
                        f"• Sample size: {n_participants} participants<br>" +
                        "• Representative sample from target population<br>" +
                        "• Clear inclusion/exclusion criteria",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "2. Single Time Point Assessment":
                # Draw timeline with single time point marker
                timeline_y = center_y
                
                fig.add_shape(
                    type="line",
                    x0=100, x1=700,
                    y0=timeline_y, y1=timeline_y,
                    line=dict(color="black", width=2),
                )
                
                # Add single time point marker
                time_point_x = 400
                
                fig.add_shape(
                    type="line",
                    x0=time_point_x, x1=time_point_x,
                    y0=timeline_y - 20, y1=timeline_y + 20,
                    line=dict(color="red", width=3),
                )
                
                fig.add_annotation(
                    x=time_point_x, y=timeline_y + 40,
                    text="Single Point in Time Assessment",
                    showarrow=False,
                    font=dict(size=16, color="red")
                )
                
                # Add participant sampling visual
                fig.add_shape(
                    type="rect",
                    x0=time_point_x - 100, x1=time_point_x + 100,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(100, 149, 237, 0.3)"
                )
                
                fig.add_annotation(
                    x=time_point_x, y=timeline_y - 100,
                    text=f"Participants<br>n={n_participants}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Connect time point to participants
                fig.add_shape(
                    type="line",
                    x0=time_point_x, x1=time_point_x,
                    y0=timeline_y + 20, y1=timeline_y - 70,
                    line=dict(color="black", width=1, dash="dash"),
                )
                
                # Add explanation about cross-sectional timing
                explanation_y = timeline_y + 100
                
                fig.add_shape(
                    type="rect",
                    x0=center_x - 210, x1=center_x + 220,
                    y0=explanation_y - 30, y1=explanation_y + 120,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=center_x, y=explanation_y+50,
                    text="Step 2: Collect data at a single point in time<br>" +
                        "• No follow-up period unlike cohort studies<br>" +
                        "• Snapshot of disease and exposure status<br>" +
                        "• Cost and time efficient design",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add contrast with longitudinal studies
                fig.add_annotation(
                    x=150, y=timeline_y - 20,
                    text="Past",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                fig.add_annotation(
                    x=650, y=timeline_y - 20,
                    text="Future",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Contrast info box
                contrast_y = timeline_y - 200
                
                fig.add_shape(
                    type="rect",
                    x0=center_x - 230, x1=center_x + 230,
                    y0=contrast_y - 70, y1=contrast_y + 20,
                    line=dict(color="orange", width=1),
                    fillcolor="rgba(255, 229, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=center_x, y=contrast_y-20,
                    text="❗ Unlike cohort and case-control studies,<br>" +
                        "cross-sectional designs cannot establish<br>" +
                        "temporal relationships between exposure and disease",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
            elif steps == "3. Measure Exposure & Disease":
                # Create a 2x2 grid showing the four possible combinations
                grid_center_x = center_x
                grid_center_y = center_y
                
                cell_width = 150
                cell_height = 150
                spacing = 20
                
                # Cell colors
                colors = {
                    (0, 0): 'rgba(220, 220, 220, 0.5)',  # Unexposed, No Disease
                    (0, 1): 'rgba(255, 100, 100, 0.5)',  # Unexposed, Disease
                    (1, 0): 'rgba(100, 100, 255, 0.5)',  # Exposed, No Disease
                    (1, 1): 'rgba(160, 32, 240, 0.5)'    # Exposed, Disease
                }
                
                # Calculate counts and percentages for each cell
                exposed_no_disease = exposed_total - exposed_with_disease
                unexposed_no_disease = unexposed_total - unexposed_with_disease
                
                cell_counts = {
                    (0, 0): unexposed_no_disease,
                    (0, 1): unexposed_with_disease,
                    (1, 0): exposed_no_disease,
                    (1, 1): exposed_with_disease
                }
                
                cell_percentages = {
                    (0, 0): unexposed_no_disease / n_participants,
                    (0, 1): unexposed_with_disease / n_participants,
                    (1, 0): exposed_no_disease / n_participants,
                    (1, 1): exposed_with_disease / n_participants
                }
                
                # Cell positions
                positions = {
                    (0, 0): (grid_center_x - cell_width - spacing/2, grid_center_y - cell_height - spacing/2),
                    (0, 1): (grid_center_x - cell_width - spacing/2, grid_center_y + spacing/2),
                    (1, 0): (grid_center_x + spacing/2, grid_center_y - cell_height - spacing/2),
                    (1, 1): (grid_center_x + spacing/2, grid_center_y + spacing/2)
                }
                
                # Draw cells
                for pos, (x, y) in positions.items():
                    fig.add_shape(
                        type="rect",
                        x0=x, x1=x + cell_width,
                        y0=y, y1=y + cell_height,
                        fillcolor=colors[pos],
                        line=dict(color="black", width=1)
                    )
                    
                    # Add cell label and count
                    fig.add_annotation(
                        x=x + cell_width/2, y=y + cell_height/2,
                        text=f"n={cell_counts[pos]}<br>({cell_percentages[pos]:.1%})",
                        showarrow=False,
                        font=dict(size=14)
                    )
                
                # Add header labels
                # Exposure headers
                fig.add_annotation(
                    x=grid_center_x - cell_width/2 - spacing/2, y=grid_center_y - cell_height - spacing/2 - 40,
                    text="Unexposed",
                    showarrow=False,
                    font=dict(size=16)
                )
                
                fig.add_annotation(
                    x=grid_center_x + cell_width/2 + spacing/2, y=grid_center_y - cell_height - spacing/2 - 40,
                    text="Exposed",
                    showarrow=False,
                    font=dict(size=16)
                )
                
                # Disease headers
                fig.add_annotation(
                    x=grid_center_x - cell_width - spacing/2 - 40, y=grid_center_y - cell_height/2 - spacing/2,
                    text="No Disease",
                    showarrow=False,
                    font=dict(size=16),
                    textangle=90
                )
                
                fig.add_annotation(
                    x=grid_center_x - cell_width - spacing/2 - 40, y=grid_center_y + cell_height/2 + spacing/2,
                    text="Disease",
                    showarrow=False,
                    font=dict(size=16),
                    textangle=90
                )
                
                # Add explanation
                explanation_y = grid_center_y - 200
                

                fig.add_annotation(
                    x=center_x, y=explanation_y -100,
                    text="Step 3: Simultaneously assess exposure and disease<br>" +
                        f"• Overall disease prevalence: {overall_prevalence:.1%}<br>" +
                        f"• Disease in exposed: {prevalence_exposed:.1%}<br>" +
                        f"• Disease in unexposed: {prevalence_unexposed:.1%}",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "4. Analysis & Interpretation":
                # Create results visualization
                results_y = center_y
                
                # Main results box
                fig.add_shape(
                    type="rect",
                    x0=center_x - 200, x1=center_x + 200,
                    y0=results_y - 160, y1=results_y + 120,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.1)"
                )
                
                # Add title
                fig.add_annotation(
                    x=center_x, y=results_y + 90,
                    text="<b>Cross-sectional Analysis Results</b>",
                    showarrow=False,
                    font=dict(size=16)
                )
                
                # Add main results
                results_text = (
                    f"Disease Prevalence:<br>" +
                    f"• Overall: {overall_prevalence:.1%}<br>" +
                    f"• In Exposed: {prevalence_exposed:.1%}<br>" +
                    f"• In Unexposed: {prevalence_unexposed:.1%}<br><br>" +
                    f"Measures of Association:<br>" +
                    f"• Prevalence Ratio (PR): {prevalence_ratio:.2f}<br>" +
                    f"• Prevalence Odds Ratio (POR): {odds_ratio:.2f}"
                )
                
                fig.add_annotation(
                    x=center_x, y=results_y-40,
                    text=results_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
                
                # Add interpretation box
                interpretation_y = results_y - 200
                
                fig.add_shape(
                    type="rect",
                    x0=center_x - 300, x1=center_x + 300,
                    y0=interpretation_y - 100, y1=interpretation_y + 20,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                # Determine interpretation text
                if prevalence_ratio > 1.5:
                    interpretation = f"The prevalence of disease is {prevalence_ratio:.1f} times higher in the exposed group."
                    strength = "strong"
                elif prevalence_ratio > 1.2:
                    interpretation = f"There is a moderate association between exposure and disease (PR={prevalence_ratio:.2f})."
                    strength = "moderate"
                elif prevalence_ratio > 0.8:
                    interpretation = f"There is little or no association between exposure and disease (PR={prevalence_ratio:.2f})."
                    strength = "weak or no"
                else:
                    interpretation = f"The exposure appears to be associated with lower disease prevalence (PR={prevalence_ratio:.2f})."
                    strength = "protective"
                
                explanation_text = (
                    f"Step 4: Analyze and interpret results<br>" +
                    f"• {interpretation}<br>" +
                    f"• A {strength} association is observed.<br>" +
                    f"• CAUTION: Cannot establish causality due to cross-sectional nature"
                )
                
                fig.add_annotation(
                    x=center_x, y=interpretation_y-40,
                    text=explanation_text,
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
                # Add limitations box
                limitations_y = results_y + 200
                
                fig.add_shape(
                    type="rect",
                    x0=center_x - 280, x1=center_x + 280,
                    y0=limitations_y - 30, y1=limitations_y + 100,
                    line=dict(color="red", width=1),
                    fillcolor="rgba(255, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=center_x, y=limitations_y+30,
                    text="<b>Key Limitations of Cross-sectional Design</b><br>" +
                        "• Cannot establish temporal sequence (chicken-egg problem)<br>" +
                        "• Prevalence-incidence bias for diseases of short duration<br>" +
                        "• Survival bias for fatal diseases",
                    showarrow=False,
                    font=dict(size=14),
                    align="center"
                )
            
            # Update layout
            fig.update_layout(
                title="Cross-sectional Study Design",
                height=total_height,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, total_width]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, total_height]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Conceptual overview tab
        with tab_concepts:
            st.subheader("Cross-sectional Study: Conceptual Overview")
            
            st.write("""
            Cross-sectional studies examine the relationship between variables of interest within a population 
            at a single point in time, similar to taking a snapshot. These studies are particularly useful 
            for determining prevalence and identifying associations.
            """)
            
            # Create a two-column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Key Strengths")
                st.write("""
                1. **Efficient**: Quick to conduct with no follow-up period
                2. **Resource-friendly**: Generally less expensive than longitudinal designs
                3. **Multiple outcomes**: Can study many variables simultaneously
                4. **Prevalence data**: Excellent for estimating disease prevalence
                """)
            
            with col2:
                st.write("#### Limitations")
                st.write("""
                1. **Temporality**: Cannot establish cause-effect relationships
                2. **Prevalence-incidence bias**: Misses diseases of short duration
                3. **Survival bias**: Underrepresents fatal conditions
                4. **Recall bias**: May affect exposure assessment
                """)
            
            # Create a simple process flow diagram
            process_fig = go.Figure()
            steps = ["Define Population", "Design Survey", "Collect Data", "Analyze", "Interpret"]
            x_positions = [100, 250, 400, 550, 700]
            y_position = 100
            
            # Draw process boxes
            for i, (step, x_pos) in enumerate(zip(steps, x_positions)):
                # Draw box
                process_fig.add_shape(
                    type="rect",
                    x0=x_pos-70, x1=x_pos+70,
                    y0=y_position-30, y1=y_position+30,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(100, 149, 237, 0.3)"
                )
                
                # Add step label
                process_fig.add_annotation(
                    x=x_pos, y=y_position,
                    text=f"{i+1}. {step}",
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                # Add connecting line (except for the last step)
                if i < len(steps) - 1:
                    next_x = x_positions[i+1]
                    process_fig.add_shape(
                        type="line",
                        x0=x_pos+70, x1=next_x-70,
                        y0=y_position, y1=y_position,
                        line=dict(color="blue", width=1),
                    )
            
            # Update layout
            process_fig.update_layout(
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[50, 150]
                )
            )
            
            st.plotly_chart(process_fig, use_container_width=True)
            
            # "Snapshot" concept visualization
            st.subheader("The 'Snapshot' Concept")
            
            snapshot_fig = go.Figure()
            
            # Draw timeline
            snapshot_fig.add_shape(
                type="line",
                x0=100, x1=700,
                y0=100, y1=100,
                line=dict(color="black", width=2),
            )
            
            # Draw time periods
            time_periods = ["Past", "Present", "Future"]
            x_positions = [200, 400, 600]
            
            for period, x_pos in zip(time_periods, x_positions):
                snapshot_fig.add_annotation(
                    x=x_pos, y=80,
                    text=period,
                    showarrow=False,
                    font=dict(size=14)
                )
            
            # Add study type time periods
            # Cross-sectional study (point)
            snapshot_fig.add_shape(
                type="line",
                x0=400, x1=400,
                y0=90, y1=110,
                line=dict(color="red", width=4),
            )
            
            snapshot_fig.add_annotation(
                x=400, y=130,
                text="Cross-sectional Study<br>(Single time point)",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            
            # Case-control study (retrospective)
            snapshot_fig.add_shape(
                type="line",
                x0=400, x1=200,
                y0=160, y1=160,
                line=dict(color="blue", width=2, dash="dash"),
            )
            
            snapshot_fig.add_annotation(
                x=300, y=180,
                text="Case-Control Study<br>(Retrospective)",
                showarrow=False,
                font=dict(size=12, color="blue")
            )
            
            # Cohort study (prospective)
            snapshot_fig.add_shape(
                type="line",
                x0=300, x1=600,
                y0=210, y1=210,
                line=dict(color="green", width=2),
            )
            
            snapshot_fig.add_annotation(
                x=450, y=230,
                text="Cohort Study<br>(Prospective)",
                showarrow=False,
                font=dict(size=12, color="green")
            )
            
            # Update layout
            snapshot_fig.update_layout(
                height=300,
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[50, 750]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[50, 250]
                )
            )
            
            st.plotly_chart(snapshot_fig, use_container_width=True)
        
        # Interactive exploration tab
        with tab_interactive:
            st.subheader("Interactive Prevalence Explorer")
            
            # Let user adjust parameters
            col1, col2 = st.columns(2)
            
            with col1:
                custom_disease_prev = st.slider(
                    "Disease Prevalence", 
                    0.05, 0.5, 0.2, 
                    key="custom_disease_prev"
                )
                custom_exposure_prev = st.slider(
                    "Exposure Prevalence", 
                    0.1, 0.7, 0.3, 
                    key="custom_exposure_prev"
                )
            
            with col2:
                custom_pr = st.slider(
                    "Prevalence Ratio", 
                    0.5, 5.0, 2.0, 
                    step=0.1,
                    key="custom_pr"
                )
                custom_n = st.slider(
                    "Sample Size", 
                    100, 2000, 500, 
                    step=100,
                    key="custom_n"
                )
            
            # Calculate expected counts and prevalences
            # First, calculate unexposed prevalence
            prob_unexposed = custom_disease_prev / (custom_exposure_prev * custom_pr + (1 - custom_exposure_prev))
            prob_exposed = prob_unexposed * custom_pr
            
            # Calculate counts
            exposed_n = int(custom_n * custom_exposure_prev)
            unexposed_n = custom_n - exposed_n
            
            exposed_cases = int(exposed_n * prob_exposed)
            unexposed_cases = int(unexposed_n * prob_unexposed)
            
            total_cases = exposed_cases + unexposed_cases
            
            # Calculate observed prevalence ratio and odds ratio
            obs_prev_exposed = exposed_cases / exposed_n
            obs_prev_unexposed = unexposed_cases / unexposed_n
            obs_pr = obs_prev_exposed / obs_prev_unexposed
            
            odds_exposed = exposed_cases / (exposed_n - exposed_cases)
            odds_unexposed = unexposed_cases / (unexposed_n - unexposed_cases)
            obs_or = odds_exposed / odds_unexposed
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Exposed Group", f"{exposed_n} participants")
                st.metric("Disease in Exposed", f"{exposed_cases} ({obs_prev_exposed:.1%})")
            
            with col2:
                st.metric("Unexposed Group", f"{unexposed_n} participants")
                st.metric("Disease in Unexposed", f"{unexposed_cases} ({obs_prev_unexposed:.1%})")
            
            with col3:
                st.metric("Observed Prevalence Ratio", f"{obs_pr:.2f}")
                st.metric("Observed Odds Ratio", f"{obs_or:.2f}")
                st.metric("Overall Prevalence", f"{total_cases/custom_n:.1%}")
            
            # Create 2x2 table visualization
            st.subheader("Interactive 2×2 Table")
            
            # Create the visualization
            table_fig = go.Figure()
            
            # Create a 2x2 table
            table_width = 400
            table_height = 300
            
            # Draw table outline
            table_fig.add_shape(
                type="rect",
                x0=0, x1=table_width,
                y0=0, y1=table_height,
                line=dict(color="black", width=2),
                fillcolor="white"
            )
            
            # Draw inner lines
            # Vertical divider
            table_fig.add_shape(
                type="line",
                x0=table_width/2, x1=table_width/2,
                y0=0, y1=table_height,
                line=dict(color="black", width=1),
            )
            
            # Horizontal divider
            table_fig.add_shape(
                type="line",
                x0=0, x1=table_width,
                y0=table_height/2, y1=table_height/2,
                line=dict(color="black", width=1),
            )
            
            # Add headers
            table_fig.add_annotation(
                x=table_width/4, y=table_height + 30,
                text="Exposed",
                showarrow=False,
                font=dict(size=14)
            )
            
            table_fig.add_annotation(
                x=3*table_width/4, y=table_height + 30,
                text="Unexposed",
                showarrow=False,
                font=dict(size=14)
            )
            
            table_fig.add_annotation(
                x=-60, y=3*table_height/4,
                text="No Disease",
                showarrow=False,
                font=dict(size=14)
            )
            
            table_fig.add_annotation(
                x=-60, y=table_height/4,
                text="Disease",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Add cell values with cell shading
            # Cell a: Exposed, No Disease
            exposed_no_disease = exposed_n - exposed_cases
            
            table_fig.add_shape(
                type="rect",
                x0=1, x1=table_width/2 - 1,
                y0=table_height/2 + 1, y1=table_height - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(100, 100, 255, 0.15)"
            )
            
            table_fig.add_annotation(
                x=table_width/4, y=3*table_height/4,
                text=f"{exposed_no_disease}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Cell b: Unexposed, No Disease
            unexposed_no_disease = unexposed_n - unexposed_cases
            
            table_fig.add_shape(
                type="rect",
                x0=table_width/2 + 1, x1=table_width - 1,
                y0=table_height/2 + 1, y1=table_height - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(220, 220, 220, 0.15)"
            )
            
            table_fig.add_annotation(
                x=3*table_width/4, y=3*table_height/4,
                text=f"{unexposed_no_disease}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Cell c: Exposed, Disease
            table_fig.add_shape(
                type="rect",
                x0=1, x1=table_width/2 - 1,
                y0=1, y1=table_height/2 - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(160, 32, 240, 0.15)"
            )
            
            table_fig.add_annotation(
                x=table_width/4, y=table_height/4,
                text=f"{exposed_cases}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Cell d: Unexposed, Disease
            table_fig.add_shape(
                type="rect",
                x0=table_width/2 + 1, x1=table_width - 1,
                y0=1, y1=table_height/2 - 1,
                line=dict(color="black", width=1),
                fillcolor="rgba(255, 100, 100, 0.15)"
            )
            
            table_fig.add_annotation(
                x=3*table_width/4, y=table_height/4,
                text=f"{unexposed_cases}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Add row and column totals
            # Exposed total
            table_fig.add_annotation(
                x=table_width/4, y=-30,
                text=f"Total: {exposed_n}",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Unexposed total
            table_fig.add_annotation(
                x=3*table_width/4, y=-30,
                text=f"Total: {unexposed_n}",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Disease total
            table_fig.add_annotation(
                x=table_width + 60, y=table_height/4,
                text=f"Total: {total_cases}",
                showarrow=False,
                font=dict(size=12)
            )
            
            # No disease total
            table_fig.add_annotation(
                x=table_width + 60, y=3*table_height/4,
                text=f"Total: {custom_n - total_cases}",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Add measures of association
            table_fig.add_annotation(
                x=table_width/2, y=-80,
                text=f"Prevalence Ratio (PR) = {obs_pr:.2f}<br>Prevalence Odds Ratio (POR) = {obs_or:.2f}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Update layout
            table_fig.update_layout(
                title="Cross-sectional 2×2 Table",
                height=600,
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-100, table_width + 100]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-100, table_height + 50]
                )
            )
            
            st.plotly_chart(table_fig, use_container_width=True)
            
            # Add a visualization of PR interpretation
            st.subheader("Prevalence Ratio Interpretation")
            
            pr_fig = go.Figure()
            
            # Create a scale visualization
            scale_width = 600
            center_x = scale_width / 2
            
            # Draw baseline (PR = 1)
            pr_fig.add_shape(
                type="line",
                x0=0, x1=scale_width,
                y0=50, y1=50,
                line=dict(color="black", width=2),
            )
            
            # Draw tick marks
            ticks = [0.25, 0.5, 1, 2, 4]
            positions = [100, 200, 300, 400, 500]
            
            for tick, pos in zip(ticks, positions):
                # Draw tick
                pr_fig.add_shape(
                    type="line",
                    x0=pos, x1=pos,
                    y0=45, y1=55,
                    line=dict(color="black", width=1),
                )
                
                # Add label
                pr_fig.add_annotation(
                    x=pos, y=35,
                    text=str(tick),
                    showarrow=False,
                    font=dict(size=12)
                )
            
            # Add interpretation regions
            # Protective region
            pr_fig.add_shape(
                type="rect",
                x0=0, x1=300,
                y0=60, y1=100,
                line=dict(color="green", width=1),
                fillcolor="rgba(0, 200, 0, 0.1)"
            )
            
            pr_fig.add_annotation(
                x=150, y=80,
                text="Protective Association<br>PR < 1",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Harmful region
            pr_fig.add_shape(
                type="rect",
                x0=300, x1=scale_width,
                y0=60, y1=100,
                line=dict(color="red", width=1),
                fillcolor="rgba(255, 0, 0, 0.1)"
            )
            
            pr_fig.add_annotation(
                x=450, y=80,
                text="Harmful Association<br>PR > 1",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Add marker for the current PR
            pr_position = 300  # Default (PR = 1)
            
            if custom_pr < 1:
                # Scale for protective PRs
                pr_position = 300 - (300 - 100) * (1 - custom_pr) / 0.75
            else:
                # Scale for harmful PRs
                pr_position = 300 + (scale_width - 300) * min(custom_pr - 1, 3) / 3
            
            pr_fig.add_shape(
                type="line",
                x0=pr_position, x1=pr_position,
                y0=0, y1=120,
                line=dict(color="blue", width=2, dash="dash"),
            )
            
            pr_fig.add_annotation(
                x=pr_position, y=120,
                text=f"Your PR: {custom_pr:.1f}",
                showarrow=False,
                font=dict(size=14, color="blue"),
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1,
                borderpad=3
            )
            
            # Add neutral line
            pr_fig.add_shape(
                type="line",
                x0=300, x1=300,
                y0=0, y1=120,
                line=dict(color="black", width=2, dash="dot"),
            )
            
            pr_fig.add_annotation(
                x=300, y=10,
                text="No Association (PR = 1)",
                showarrow=False,
                font=dict(size=12)
            )
            
            # Update layout
            pr_fig.update_layout(
                title="Prevalence Ratio Interpretation Scale",
                height=400,
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-50, scale_width + 50]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 140]
                )
            )
            
            st.plotly_chart(pr_fig, use_container_width=True)
        
        # Design comparison tab
        with tab_compare:
            st.subheader("Study Design Comparison: Cross-sectional vs. Other Designs")
            
            # Create a 3-column layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("### Cross-sectional Study")
                st.write("""
                **Approach**: Assess exposure and disease simultaneously in a population
                
                **Strengths**:
                - Quick and inexpensive
                - No follow-up required
                - Can study multiple variables
                - Good for prevalence data
                
                **Limitations**:
                - Cannot establish temporality
                - Prevalence-incidence bias
                - Selection bias
                - Recall bias
                """)
            
            with col2:
                st.write("### Cohort Study")
                st.write("""
                **Approach**: Follow exposed and unexposed groups over time
                
                **Strengths**:
                - Establishes temporality
                - Can study multiple outcomes
                - Directly measures incidence
                - Can calculate relative risks
                
                **Limitations**:
                - Time and resource intensive
                - Loss to follow-up
                - Inefficient for rare diseases
                - Selection bias at enrollment
                """)
            
            with col3:
                st.write("### Case-Control Study")
                st.write("""
                **Approach**: Compare past exposure in people with and without disease
                
                **Strengths**:
                - Efficient for rare diseases
                - Requires fewer subjects
                - Can study multiple exposures
                - Relatively quick
                
                **Limitations**:
                - Recall bias
                - Selection of appropriate controls
                - Cannot directly measure incidence
                - Temporal relationship may be unclear
                """)
            
            # Add decision flowchart for study design selection
            st.subheader("When to Use a Cross-sectional Design")
            
            decision_fig = go.Figure()
            
            # Define decision points
            decisions = [
                {"text": "Research Question", "x": 400, "y": 500, "decision": True},
                {"text": "Are you studying prevalence?", "x": 400, "y": 400, "decision": True},
                {"text": "Is temporality important?", "x": 250, "y": 300, "decision": True},
                {"text": "Is disease rare?", "x": 550, "y": 300, "decision": True},
                {"text": "Cohort or RCT", "x": 250, "y": 200, "decision": False, "final": True},
                {"text": "Cross-sectional Study", "x": 400, "y": 200, "decision": False, "final": True},
                {"text": "Case-Control Study", "x": 550, "y": 200, "decision": False, "final": True},
            ]
            
            # Draw decision nodes and endpoints
            for decision in decisions:
                box_color = "green" if decision.get("final", False) else "blue"
                box_fill = "rgba(0, 200, 0, 0.1)" if decision.get("final", False) else "rgba(200, 200, 255, 0.1)"
                
                # Draw box
                decision_fig.add_shape(
                    type="rect",
                    x0=decision["x"] - 75, x1=decision["x"] + 75,
                    y0=decision["y"] - 25, y1=decision["y"] + 25,
                    line=dict(color=box_color, width=2),
                    fillcolor=box_fill
                )
                
                # Add text
                decision_fig.add_annotation(
                    x=decision["x"], y=decision["y"],
                    text=decision["text"],
                    showarrow=False,
                    font=dict(size=11)
                )
            
            # Connect nodes with lines
            # First level
            decision_fig.add_shape(
                type="line",
                x0=400, x1=400,
                y0=475, y1=425,
                line=dict(color="black", width=1),
            )
            
            # Second level - left and right branches
            decision_fig.add_shape(
                type="line",
                x0=400, x1=250,
                y0=375, y1=325,
                line=dict(color="black", width=1),
            )
            
            decision_fig.add_shape(
                type="line",
                x0=400, x1=550,
                y0=375, y1=325,
                line=dict(color="black", width=1),
            )
            
            # Third level to final
            decision_fig.add_shape(
                type="line",
                x0=250, x1=250,
                y0=275, y1=225,
                line=dict(color="black", width=1),
            )
            
            decision_fig.add_shape(
                type="line",
                x0=400, x1=400,
                y0=345, y1=225,
                line=dict(color="black", width=1),
            )
            
            decision_fig.add_shape(
                type="line",
                x0=550, x1=550,
                y0=275, y1=225,
                line=dict(color="black", width=1),
            )
            
            # Add Yes/No labels
            decision_fig.add_annotation(
                x=320, y=385,
                text="No",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=480, y=385,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=250, y=250,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=550, y=250,
                text="Yes",
                showarrow=False,
                font=dict(size=10)
            )
            
            decision_fig.add_annotation(
                x=350, y=275,
                text="No",
                showarrow=False,
                font=dict(size=10),
                xanchor="center"
            )
            
            # Update layout
            decision_fig.update_layout(
                height=550,
                showlegend=False,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[150, 650]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[150, 550]
                )
            )
            
            st.plotly_chart(decision_fig, use_container_width=True)
            
            # Add best use cases
            st.subheader("Best Use Cases for Cross-sectional Studies")
            
            use_cases = [
                {
                    "title": "Disease Surveillance",
                    "description": "Monitoring prevalence of diseases in populations",
                    "icon": "📊"
                },
                {
                    "title": "Health Service Planning",
                    "description": "Understanding healthcare needs based on disease burden",
                    "icon": "🏥"
                },
                {
                    "title": "Hypothesis Generation",
                    "description": "Identifying associations that can be tested in future studies",
                    "icon": "💡"
                },
                {
                    "title": "Multiple Risk Factors",
                    "description": "Studying numerous variables simultaneously",
                    "icon": "🔍"
                },
                {
                    "title": "Resources Constraints",
                    "description": "When time and budget are limited",
                    "icon": "💰"
                }
            ]
            
            # Create use case cards
            for i in range(0, len(use_cases), 3):
                cols = st.columns(min(3, len(use_cases) - i))
                for j in range(min(3, len(use_cases) - i)):
                    case = use_cases[i+j]
                    with cols[j]:
                        st.markdown(f"### {case['icon']} {case['title']}")
                        st.write(case['description'])
                        
    # ENHANCED NESTED CASE-CONTROL STUDY SECTION
    elif study_design == "Nested Case-Control Study":
        st.header("Nested Case-Control Study Simulation")
        
        # Create tabs for different aspects of the visualization
        tab_main, tab_concepts, tab_interactive, tab_compare = st.tabs([
            "📉 Main Visualization", 
            "📝 Conceptual Overview", 
            "📊 Interactive Exploration", 
            "⚖️ Design Comparison"
        ])
        
        # Parameters for cohort simulation (shared across all tabs)
        with tab_main:
            col1, col2 = st.columns(2)
            with col1:
                n_cohort = st.slider("Initial Cohort Size", 1000, 10000, 5000, key="ncc_cohort_size")
                followup_years = st.slider("Follow-up Years", 1, 10, 5, key="ncc_followup")
            with col2:
                n_controls = st.slider("Number of Controls per Case", 1, 5, 4, key="ncc_controls")
                exposure_effect = st.slider("Exposure Effect (Hazard Ratio)", 1.0, 5.0, 2.0, key="ncc_effect")
        
        # Function to generate cohort data
        def generate_nested_cc_data(n_cohort, followup_years, exposure_effect):
            # Generate baseline data
            data = pd.DataFrame({
                'id': range(n_cohort),
                'exposure': np.random.binomial(1, 0.3, n_cohort),
                'age': np.random.normal(50, 10, n_cohort),
                'sex': np.random.binomial(1, 0.5, n_cohort)
            })
            
            # Generate survival times
            baseline_hazard = 0.1
            lambda_i = baseline_hazard * np.exp(np.log(exposure_effect) * data['exposure'])
            data['time_to_event'] = np.random.exponential(1/lambda_i)
            data['time_to_censoring'] = np.random.uniform(0, followup_years, n_cohort)
            
            # Determine observed time and case status
            data['observed_time'] = np.minimum(data['time_to_event'], 
                                              np.minimum(data['time_to_censoring'], followup_years))
            data['is_case'] = (data['time_to_event'] <= data['time_to_censoring']) & \
                             (data['time_to_event'] <= followup_years)
            
            return data
        
        # Generate data once to be used in all tabs
        np.random.seed(42)
        cohort_data = generate_nested_cc_data(n_cohort, followup_years, exposure_effect)
        cases = cohort_data[cohort_data['is_case']].copy()
        
        # Main visualization tab
        with tab_main:
            # Step-by-step progression slider
            steps = st.radio(
                "Study Design Stages:",
                ["1. Initial Cohort", "2. Follow-up Period", "3. Case Identification", 
                 "4. Control Selection", "5. Analysis"],
                horizontal=True,
                key="ncc_steps"
            )
            
            # Create dynamic visualization based on selected step
            fig = go.Figure()
            
            # Constants for visualization
            total_width = 800
            total_height = 600
            timeline_y = 350
            
            # Draw timeline
            fig.add_shape(
                type="line",
                x0=50, x1=750,
                y0=timeline_y, y1=timeline_y,
                line=dict(color="black", width=3),
            )
            
            # Add time labels
            for year in range(followup_years + 1):
                fig.add_annotation(
                    x=50 + (year * 700 / followup_years),
                    y=timeline_y - 20,
                    text=f"Year {year}",
                    showarrow=False,
                    font=dict(size=10)
                )
            
            # Different elements based on step
            if steps == "1. Initial Cohort":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=20, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=650,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=400, y=timeline_y - 100,
                    text="Step 1: Define and recruit a cohort of individuals<br>" +
                         f"• Cohort size: {n_cohort} participants<br>" +
                         "• Collect baseline information on all participants<br>" +
                         "• Participants are free of disease at baseline",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "2. Follow-up Period":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=20, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw sample of subjects being followed
                n_sample = 15  # Number of subjects to show
                sample_data = cohort_data.sample(n_sample)
                
                for i, (_, subject) in enumerate(sample_data.iterrows()):
                    y_pos = timeline_y + 40 - i * 10
                    x_end = 50 + (subject['observed_time'] / followup_years) * 700
                    
                    # Subject's follow-up line
                    fig.add_shape(
                        type="line",
                        x0=50, x1=x_end,
                        y0=y_pos, y1=y_pos,
                        line=dict(
                            color="blue" if subject['exposure'] else "gray",
                            width=2,
                            dash="solid"
                        ),
                    )
                    
                    # Add event marker if subject becomes a case
                    if subject['is_case']:
                        fig.add_trace(go.Scatter(
                            x=[x_end],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="x"),
                            name="Case",
                            showlegend=False
                        ))
                

                fig.add_annotation(
                    x=705, y=timeline_y - 100,
                    text="Step 2: Follow cohort over time<br>" +
                         f"• Follow-up period: {followup_years} years<br>" +
                         "• Monitor for disease occurrences<br>" +
                         "• Some participants may be lost to follow-up",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "3. Case Identification":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=20, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Show cases occurring over time
                # Select a few cases at different time points
                selected_cases = cases.sample(min(5, len(cases)))
                
                for i, (_, case) in enumerate(selected_cases.iterrows()):
                    y_pos = timeline_y + 20 - i * 15
                    x_pos = 50 + (case['observed_time'] / followup_years) * 700
                    
                    # Show follow-up until event
                    fig.add_shape(
                        type="line",
                        x0=50, x1=x_pos,
                        y0=y_pos, y1=y_pos,
                        line=dict(
                            color="blue" if case['exposure'] else "gray",
                            width=2,
                            dash="solid"
                        ),
                    )
                    
                    # Add case marker
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(color="red", size=12, symbol="x"),
                        name="Case" if i == 0 else None,
                        showlegend=i == 0
                    ))
                    
                    # Add case label
                    fig.add_annotation(
                        x=x_pos, y=y_pos + 10,
                        text=f"Case at Year {case['observed_time']:.1f}",
                        showarrow=False,
                        font=dict(size=10),
                        xanchor="center",
                        yanchor="bottom"
                    )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=200, x1=600,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                n_cases = sum(cohort_data['is_case'])
                
                fig.add_annotation(
                    x=400, y=timeline_y - 100,
                    text="Step 3: Identify all cases<br>" +
                         f"• Total cases detected: {n_cases}<br>" +
                         "• Record exact time of case occurrence<br>" +
                         "• Each case will need matched controls",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "4. Control Selection":
                # Show case-control selection process for a few time points
                case_times = [1.5, 3, 4.2]  # in years
                case_positions = [50 + (t * 700 / followup_years) for t in case_times]
                
                # For each case, show the risk set selection process
                for i, (time, x_pos) in enumerate(zip(case_times, case_positions)):
                    # Case occurrence
                    case_y = 350 - i * 80
                    
                    # Add case label
                    fig.add_annotation(
                        x=x_pos, y=case_y + 30,
                        text=f"Case at Year {time:.1f}",
                        showarrow=False,
                        font=dict(size=12),
                        xanchor="center"
                    )
                    
                    # Draw case symbol
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[case_y],
                        mode="markers",
                        marker=dict(color="red", size=15, symbol="x"),
                        name="Case" if i==0 else None,
                        showlegend=i==0
                    ))
                    
                    # Draw a dotted line to show time point
                    fig.add_shape(
                        type="line",
                        x0=x_pos, x1=x_pos,
                        y0=timeline_y, y1=case_y,
                        line=dict(color="red", width=1, dash="dash"),
                    )
                    
                    # Risk set (people still being followed at this time)
                    # Calculate approximately how many would be left in risk set
                    prop_remaining = 1 - (time / followup_years) * 0.3  # Simplified estimate
                    n_risk_set = int(n_cohort * prop_remaining)
                    
                    fig.add_shape(
                        type="rect",
                        x0=x_pos-50, x1=x_pos+50,
                        y0=case_y-40, y1=case_y-10,
                        line=dict(color="blue", width=1),
                        fillcolor="rgba(0, 0, 255, 0.1)"
                    )
                    
                    fig.add_annotation(
                        x=x_pos, y=case_y-25,
                        text=f"Risk Set<br>n≈{n_risk_set}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                    
                    # Draw controls (simplified as dots)
                    for j in range(n_controls):
                        control_x = x_pos + (j-n_controls/2) * 12
                        control_y = case_y - 25
                        
                        fig.add_trace(go.Scatter(
                            x=[control_x],
                            y=[control_y],
                            mode="markers",
                            marker=dict(color="blue", size=8),
                            name="Control" if i==0 and j==0 else None,
                            showlegend=i==0 and j==0
                        ))
                    

                

                
                fig.add_annotation(
                    x=200, y=timeline_y - 150,
                    text="Step 4: Select controls for each case<br>" +
                         f"• For each case, select {n_controls} controls from the risk set<br>" +
                         "• Risk set = subjects still being followed at the time of case occurrence<br>" +
                         "• Controls must be disease-free at the time of matching",
                    showarrow=False,
                    font=dict(size=14),
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=8
                )
                
            elif steps == "5. Analysis":
                # Show final analysis step
                # Calculate number of cases and control sets
                n_cases = sum(cohort_data['is_case'])
                n_total_sample = n_cases * (1 + n_controls)
                
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=20, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Show a few case-control sets
                for i in range(3):
                    y_pos = timeline_y - 50 - i * 30
                    x_pos = 200 + i * 150
                    
                    # Draw case
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(color="red", size=12, symbol="x"),
                        name="Case" if i==0 else None,
                        showlegend=i==0
                    ))
                    
                    # Draw controls
                    for j in range(n_controls):
                        control_x = x_pos + 20
                        control_y = y_pos - 5 - j * 5
                        
                        fig.add_trace(go.Scatter(
                            x=[control_x],
                            y=[control_y],
                            mode="markers",
                            marker=dict(color="blue", size=8),
                            name="Control" if i==0 and j==0 else None,
                            showlegend=i==0 and j==0
                        ))
                
                # Draw analysis box
                fig.add_shape(
                    type="rect",
                    x0=600, x1=700,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="purple", width=2),
                    fillcolor="rgba(128, 0, 128, 0.1)",
                )
                
                fig.add_annotation(
                    x=650,
                    y=timeline_y - 100,
                    text=f"Analysis Phase<br>{n_cases} Case-Control Sets<br>Total sample: {n_total_sample}",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                
                fig.add_annotation(
                    x=150, y=timeline_y - 150,
                    text="Step 5: Analyze case-control sets<br>" +
                         "• Use conditional logistic regression<br>" +
                         "• Account for matching in analysis<br>" +
                         f"• Only need biomarkers for {n_total_sample} subjects ({n_total_sample/n_cohort:.1%} of cohort)",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
            
            # Update layout
            fig.update_layout(
                title="Nested Case-Control Study Design",
                height=total_height,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[timeline_y - 200, timeline_y + 200]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Optional animation 
            if st.checkbox("Show animated progression through follow-up", key="ncc_animation"):
                progress_bar = st.progress(0)
                current_year = st.empty()
                animation_plot = st.empty()
                animation_description = st.empty()  # Add this line for the description
                
                # Create animation
                for t in range(0, followup_years * 10 + 1):
                    time_point = t / 10
                    progress = int(100 * time_point / followup_years)
                    progress_bar.progress(progress)
                    current_year.markdown(f"### Year {time_point:.1f}")
                    
                    # Create frame for this time point
                    anim_fig = go.Figure()
                    
                    # Draw timeline
                    anim_fig.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=0, y1=0,
                        line=dict(color="black", width=2),
                    )
                    
                    # Add time marker
                    anim_fig.add_shape(
                        type="line",
                        x0=time_point, x1=time_point,
                        y0=-0.2, y1=20,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    # Add participants (sample)
                    sample_size = 20
                    sample = cohort_data.sample(sample_size)
                    
                    # Count cases and exposure status for the description
                    cases_so_far = sum((sample['is_case']) & (sample['observed_time'] <= time_point))
                    exposed_count = sum(sample['exposure'] == 1)
                    unexposed_count = sum(sample['exposure'] == 0)
                    exposed_cases = sum((sample['exposure'] == 1) & (sample['is_case']) & (sample['observed_time'] <= time_point))
                    unexposed_cases = sum((sample['exposure'] == 0) & (sample['is_case']) & (sample['observed_time'] <= time_point))
                    
                    for i, (_, subject) in enumerate(sample.iterrows()):
                        y_pos = i + 1
                        
                        # Draw line up to current time or event/censoring time
                        x_end = min(time_point, subject['observed_time'])
                        
                        anim_fig.add_shape(
                            type="line",
                            x0=0, x1=x_end,
                            y0=y_pos, y1=y_pos,
                            line=dict(
                                color="blue" if subject['exposure'] else "gray",
                                width=1.5
                            ),
                        )
                        
                        # Add case marker if applicable
                        if subject['is_case'] and subject['observed_time'] <= time_point:
                            anim_fig.add_trace(go.Scatter(
                                x=[subject['observed_time']],
                                y=[y_pos],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                name="Case",
                                showlegend=False
                            ))
                    
                    # Update layout
                    anim_fig.update_layout(
                        title=f"Cohort Follow-up at Year {time_point:.1f}",
                        xaxis_title="Follow-up Time (Years)",
                        height=500,
                        xaxis=dict(range=[-0.1, followup_years + 0.1]),
                        yaxis=dict(showticklabels=False,range=[0, sample_size + 1])
                    )
                    
                    animation_plot.plotly_chart(anim_fig, use_container_width=True)
                    
                    # Add description for nested case-control animation
                    with animation_description.container():
                        st.subheader("What's happening in this animation:")
                        
                        st.write(f"**Current time point:** Year {time_point:.1f} of {followup_years} years follow-up")
                        
                        st.write("**What you're seeing:**")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown("• <span style='color:blue; font-weight:bold;'>Blue lines:</span>", unsafe_allow_html=True)
                            st.markdown("• <span style='color:gray; font-weight:bold;'>Gray lines:</span>", unsafe_allow_html=True)
                            st.markdown("• <span style='color:red; font-weight:bold;'>Red X markers:</span>", unsafe_allow_html=True)
                            st.markdown("• <span style='color:red; font-weight:bold;'>Vertical red line:</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.write(f"Exposed participants ({exposed_count} people)")
                            st.write(f"Unexposed participants ({unexposed_count} people)")
                            st.write("Case events that have occurred")
                            st.write("Current point in follow-up time")
                        
                        st.write(f"**Cases so far:** {cases_so_far} total cases ({exposed_cases} in exposed, {unexposed_cases} in unexposed)")
                        
                        # Calculate and display odds ratio if enough cases
                        if exposed_cases > 0 and unexposed_cases > 0:
                            # For nested case-control, we'd typically select controls at time of each case
                            # This is a simplified representation
                            odds_ratio = (exposed_cases / unexposed_cases) / (exposed_count / unexposed_count)
                            st.write(f"**Estimated odds ratio:** {odds_ratio:.2f}")
                        
                        st.write("This animation demonstrates how a nested case-control study works within a cohort:")
                        st.write("1. We follow a cohort over time, just like in a cohort study")
                        st.write("2. As cases occur, we identify them with red X markers")
                        st.write("3. For each case, we would select controls from those still at risk at that time point")
                        st.write("4. This approach is more efficient than analyzing the entire cohort")
                        
                        # Add context-specific message based on time point
                        if time_point < followup_years * 0.25:
                            st.info("**Early follow-up:** Few cases have occurred. Sampling of controls would just be starting.")
                        elif time_point < followup_years * 0.75:
                            st.info("**Mid follow-up:** More cases accumulating. Controls are selected from those still at risk when each case occurs.")
                        else:
                            st.info("**Late follow-up:** Case accumulation is stabilizing. The nested case-control design maintains temporality while improving efficiency.")
                    
                    time_module.sleep(0.5)  # Control animation speed
        
        # Conceptual overview tab
        with tab_concepts:
            st.subheader("Nested Case-Control Study: Conceptual Overview")
            
            st.write("""
            ### Why Use a Nested Case-Control Design?
            
            A nested case-control study is a powerful variant of the case-control design that is conducted within a cohort study. 
            It provides several key advantages:
            """)
            
            # Create a two-column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Key Advantages")
                st.write("""
                1. **Efficiency**: Only a fraction of the cohort requires detailed exposure assessment
                2. **Cost-effectiveness**: Significantly reduces costs for expensive biomarker testing
                3. **Temporal clarity**: Preserves the temporal relationship between exposure and outcome
                4. **Matching capability**: Controls can be matched on time-at-risk and other factors
                """)
                
                # Use simple metrics to show the difference
                n_cases = sum(cohort_data['is_case'])
                n_total_sample = n_cases * (1 + n_controls)
                n_biomarker_assays = n_cohort
                cost_per_assay = 100  # Hypothetical cost
                total_cohort_cost = n_biomarker_assays * cost_per_assay
                ncc_cost = n_total_sample * cost_per_assay
                                
                st.metric("Total Cohort Size", n_cohort)
                st.metric("NCC Sample Size", n_total_sample, 
                        delta=f"-{n_cohort - n_total_sample}")
                st.metric("Percentage of Original Cohort", f"{n_total_sample/n_cohort:.1%}")
            
            with col2:
                st.write("#### Methodological Strengths")
                st.write("""
                1. **Risk set sampling**: Controls selected from those at risk when case occurs
                2. **Exposure assessment**: Measurement can be conducted after follow-up
                3. **Biomarker testing**: Can use stored biospecimens collected at baseline
                4. **Statistical validity**: Provides valid odds ratio estimates
                """)
                
                # Display cost comparison metrics
                st.metric("Full Cohort Biomarker Cost", f"${total_cohort_cost:,}")
                st.metric("NCC Biomarker Cost", f"${ncc_cost:,}", 
                        delta=f"-${total_cohort_cost - ncc_cost:,}")
                st.metric("Cost Savings", f"${total_cohort_cost - ncc_cost:,}")
            
            # Create explanatory figure
            st.subheader("The Nested Case-Control Process")
            
            # Simplified process figure
            process_fig = go.Figure()
            
            # Define process steps and positions
            steps = [
                "Define Full Cohort", 
                "Follow Cohort Over Time", 
                "Identify Cases", 
                "Select Controls from Risk Set",
                "Analyze Case-Control Sets"
            ]
            x_positions = [100, 250, 400, 550, 700]
            y_position = 100
            
            # Draw process boxes
            for i, (step, x_pos) in enumerate(zip(steps, x_positions)):
                # Draw box
                process_fig.add_shape(
                    type="rect",
                    x0=x_pos-70, x1=x_pos+70,
                    y0=y_position-30, y1=y_position+30,
                    line=dict(color="blue", width=2),
                    fillcolor="rgba(100, 149, 237, 0.3)"
                )
                
                # Add step label
                process_fig.add_annotation(
                    x=x_pos, y=y_position,
                    text=f"{i+1}. {step}",
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                # Add connecting arrow (except for the last step)
                if i < len(steps) - 1:
                    next_x = x_positions[i+1]
                    process_fig.add_annotation(
                        x=x_pos+80, y=y_position,
                        ax=next_x-80, ay=y_position,
                        text="",
                        showarrow=True,
                        arrowhead=2,
                        arrowwidth=1.5,
                        arrowcolor="blue"
                    )
            
            # Add specific annotations for clarity
            annotations = [
                {"x": 100, "y": 50, "text": "n = full cohort size"},
                {"x": 400, "y": 50, "text": "n = number of cases"},
                {"x": 550, "y": 50, "text": f"n = cases × {n_controls} controls"},
                {"x": 700, "y": 50, "text": f"n = {n_total_sample} (~ {n_total_sample/n_cohort:.1%} of cohort)"}
            ]
            
            for anno in annotations:
                process_fig.add_annotation(
                    x=anno["x"], y=anno["y"],
                    text=anno["text"],
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
            
            # Update layout
            process_fig.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 800]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 150]
                )
            )
            
            st.plotly_chart(process_fig, use_container_width=True)
            
            # Add explanation of risk set sampling
            st.subheader("Risk Set Sampling: The Key Concept")
            
            st.write("""
            The distinctive feature of nested case-control studies is **risk set sampling**:
            
            - For each case, controls are selected from all cohort members still at risk at the time the case occurs
            - This maintains the proper time relationship between exposure and outcome
            - Individuals who become cases later can serve as controls for earlier cases
            - The odds ratio from a nested case-control study approximates the hazard ratio from the full cohort
            """)
            
            # Add comparison to traditional case-control studies
            st.write("""
            ##### Comparison to Traditional Case-Control Studies
            
            | Feature | Nested Case-Control | Traditional Case-Control |
            | --- | --- | --- |
            | Sampling frame | Defined cohort | General population |
            | Temporal sequence | Preserved | May be unclear |
            | Exposure assessment | Can be done after follow-up | Subject to recall bias |
            | Controls | Selected from risk set | Selected from various sources |
            | Analysis | Conditional logistic regression | Unconditional logistic regression |
            | Interpretation | Approximates hazard ratio | Odds ratio only |
            """)
        
        # Interactive risk sets tab
        with tab_interactive:
            st.subheader("Interactive Risk Set Exploration")
            
            # Allow user to select time point with a slider
            selected_time = st.slider(
                "Select a time point (years):",
                min_value=0.5,
                max_value=followup_years - 0.5,
                value=2.0,
                step=0.5,
                key="ncc_time_slider"
            )
            
            # Create sample dataset for visualization (smaller than full cohort for clarity)
            np.random.seed(42)
            
            # Create a sample cohort with some attrition over time
            sample_size = 20
            sample_cohort = pd.DataFrame({
                'id': range(1, sample_size + 1),
                'event_time': np.random.exponential(scale=followup_years, size=sample_size),
                'censoring_time': np.random.uniform(followup_years/2, followup_years*1.2, size=sample_size),
                'exposure': np.random.binomial(1, 0.4, size=sample_size)
            })
            
            sample_cohort['observed_time'] = np.minimum(sample_cohort['event_time'], sample_cohort['censoring_time'])
            sample_cohort['event'] = sample_cohort['event_time'] <= sample_cohort['censoring_time']
            
            # Find who is in the risk set at the selected time
            risk_set = sample_cohort[sample_cohort['observed_time'] >= selected_time]
            
            # If there's an event at almost exactly this time, mark it as a case
            case = None
            potential_cases = sample_cohort[
                (sample_cohort['event']) & 
                (sample_cohort['event_time'] >= selected_time - 0.1) & 
                (sample_cohort['event_time'] <= selected_time + 0.1)
            ]
            
            if not potential_cases.empty:
                case = potential_cases.iloc[0]
            
            # Create a visualization of the risk set
            fig = go.Figure()
            
            # Draw timeline
            fig.add_shape(
                type="line",
                x0=0, x1=followup_years,
                y0=0, y1=0,
                line=dict(color="black", width=2)
            )
            
            # Add tick marks for each year
            for year in range(followup_years + 1):
                fig.add_shape(
                    type="line",
                    x0=year, x1=year,
                    y0=-0.2, y1=0.2,
                    line=dict(color="black", width=1)
                )
                
                fig.add_annotation(
                    x=year, y=-0.5,
                    text=f"Year {year}",
                    showarrow=False,
                    font=dict(size=10)
                )
            
            # Draw each participant
            for i, (_, subject) in enumerate(sample_cohort.iterrows()):
                y_pos = 1 + i * 0.8
                
                # Line representing follow-up
                fig.add_shape(
                    type="line",
                    x0=0, x1=subject['observed_time'],
                    y0=y_pos, y1=y_pos,
                    line=dict(
                        color="blue" if subject['exposure'] else "gray",
                        width=2 if subject['id'] in risk_set['id'].values else 1,
                        dash="solid" if subject['id'] in risk_set['id'].values else "dot"
                    )
                )
                
                # Event marker
                if subject['event']:
                    fig.add_trace(go.Scatter(
                        x=[subject['event_time']],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(
                            color="red", 
                            size=10, 
                            symbol="x"
                        ),
                        showlegend=False
                    ))
                else:
                    # Censoring marker
                    fig.add_trace(go.Scatter(
                        x=[subject['observed_time']],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(
                            color="black", 
                            size=8, 
                            symbol="circle-open"
                        ),
                        showlegend=False
                    ))
                
                # Add ID labels
                fig.add_annotation(
                    x=-0.5, y=y_pos,
                    text=f"ID: {subject['id']}",
                    showarrow=False,
                    font=dict(size=10),
                    xanchor="right"
                )
                
                # Add exposure labels
                fig.add_annotation(
                    x=-0.2, y=y_pos,
                    text="E+" if subject['exposure'] else "E-",
                    showarrow=False,
                    font=dict(
                        size=10,
                        color="blue" if subject['exposure'] else "gray"
                    ),
                    xanchor="right"
                )
            
            # Highlight the selected time
            fig.add_shape(
                type="line",
                x0=selected_time, x1=selected_time,
                y0=-1, y1=sample_size+1,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=selected_time, y=sample_size+1.5,
                text=f"Selected Time: Year {selected_time}",
                showarrow=False,
                font=dict(size=12, color="red"),
                bgcolor="white",
                bordercolor="red",
                borderwidth=1,
                borderpad=6  # Add padding
            )
            
            # Update layout
            fig.update_layout(
                title="Nested Case-Control: Risk Set at Selected Time",
                xaxis_title="Follow-up Time (Years)",
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-1, sample_size+2]
                ),
                plot_bgcolor="white",
                height=600,
                xaxis=dict(range=[-1, followup_years + 0.5])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show risk set information
            st.subheader(f"Risk Set at Year {selected_time}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Total sample members:** {len(sample_cohort)}")
                st.write(f"**Members in risk set:** {len(risk_set)}")
                st.write(f"**Risk set percentage:** {len(risk_set)/len(sample_cohort):.1%}")
                
                if case is not None:
                    st.write(f"**Case detected:** ID {case['id']}")
                    
                    # Sample controls from risk set
                    eligible_controls = risk_set[risk_set['id'] != case['id']]
                    
                    if len(eligible_controls) >= n_controls:
                        selected_controls = eligible_controls.sample(n_controls)
                        st.write(f"**Controls selected:** {', '.join([f'ID {id}' for id in selected_controls['id']])}")
                    else:
                        st.write(f"**Insufficient controls in risk set.** Only {len(eligible_controls)} available.")
                else:
                    st.write("**No case at this time point.**")
            
            with col2:
                # Display exposure status in risk set
                exposed_count = sum(risk_set['exposure'])
                unexposed_count = len(risk_set) - exposed_count
                
                exposure_data = pd.DataFrame({
                    'Exposure Status': ['Exposed', 'Unexposed'],
                    'Count': [exposed_count, unexposed_count]
                })
                
                fig = px.pie(
                    exposure_data,
                    values='Count',
                    names='Exposure Status',
                    title='Exposure Distribution in Risk Set',
                    color_discrete_map={'Exposed': 'blue', 'Unexposed': 'gray'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Control sampling simulator
            st.subheader("Control Sampling Simulation")
            
            if case is not None:
                st.write(f"### Case occurred at year {case['event_time']:.2f}")
                
                # Simulate sampling multiple times
                if st.button("Simulate control selection", key="ncc_sim_button"):
                    eligible_controls = risk_set[risk_set['id'] != case['id']]
                    
                    # Create multiple samples to show variability
                    n_simulations = 5
                    sim_results = []
                    
                    for i in range(n_simulations):
                        if len(eligible_controls) >= n_controls:
                            controls = eligible_controls.sample(n_controls)
                            exposures = sum(controls['exposure'])
                            sim_results.append({
                                'Simulation': i+1,
                                'Controls': ', '.join([f'ID {id}' for id in controls['id']]),
                                'Exposed Controls': exposures,
                                'Unexposed Controls': n_controls - exposures
                            })
                    
                    # Display results
                    if sim_results:
                        sim_df = pd.DataFrame(sim_results)
                        st.dataframe(sim_df)
                        
                        st.write("""
                        **Note how the specific controls selected may change, but the exposure distribution remains relatively stable.**
                        This is a key feature of risk set sampling - it provides valid estimates of the exposure-disease relationship.
                        """)
                    else:
                        st.write("Insufficient controls available in the risk set.")
            else:
                st.write("Select a time point where a case occurs to simulate control selection.")
        
        # Design comparison tab
        with tab_compare:
            st.subheader("Study Design Comparison: Nested Case-Control vs. Other Designs")
            
            # Create a side-by-side comparison of different study designs
            def design_comparison():
                # Create a 3-column layout
                col1, col2, col3 = st.columns(3)
                
                # Calculate key metrics
                n_cases = sum(cohort_data['is_case'])
                n_ncc_sample = n_cases * (1 + n_controls)
                subcohort_fraction = 0.2
                n_subcohort = int(n_cohort * subcohort_fraction)
                n_cases_not_in_subcohort = int(n_cases * (1 - subcohort_fraction))
                n_cc_sample = n_subcohort + n_cases_not_in_subcohort
                
                with col1:
                    st.write("### Full Cohort")
                    # Calculate metrics for full cohort
                    
                    # Create metrics display
                    st.metric("Sample Size", n_cohort)
                    st.metric("Data Collection Cost", "High", delta="Baseline")
                    st.metric("Statistical Efficiency", "100%", delta="Baseline")
                    
                    # Create simplified visualization
                    fig1 = go.Figure()
                    
                    # Draw timeline
                    fig1.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=50, y1=50,
                        line=dict(color="black", width=2)
                    )
                    
                    # Draw subjects (simplified)
                    y_positions = np.linspace(60, 200, 20)
                    
                    for i, y in enumerate(y_positions):
                        # Draw follow-up line
                        subject_time = np.random.uniform(1, followup_years)
                        is_case = np.random.random() < 0.2
                        
                        fig1.add_shape(
                            type="line",
                            x0=0, x1=subject_time,
                            y0=y, y1=y,
                            line=dict(color="gray", width=1.5)
                        )
                        
                        # Add case marker if applicable
                        if is_case:
                            fig1.add_trace(go.Scatter(
                                x=[subject_time],
                                y=[y],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                showlegend=False
                            ))
                    
                    fig1.update_layout(
                        height=400,
                        showlegend=False,
                        title="Full Cohort Follow-up",
                        xaxis=dict(title="Time (years)"),
                        yaxis=dict(showticklabels=False)
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.write("### Nested Case-Control")
                    
                    # Create metrics display
                    st.metric("Sample Size", n_ncc_sample, delta=f"{-round((1-n_ncc_sample/n_cohort)*100)}%")
                    st.metric("Data Collection Cost", "Medium", delta="Lower")
                    st.metric("Statistical Efficiency", "~95%", delta="-5%")
                    
                    # Create simplified visualization
                    fig2 = go.Figure()
                    
                    # Draw timeline
                    fig2.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=50, y1=50,
                        line=dict(color="black", width=2)
                    )
                    
                    # Draw subjects (simplified with focus on case-control selection)
                    # Similar to fig1 but highlight control selection
                    case_times = [1.5, 3, 4.2]
                    
                    for i, case_time in enumerate(case_times):
                        y_case = 80 + i*30
                        
                        # Draw case
                        fig2.add_trace(go.Scatter(
                            x=[case_time],
                            y=[y_case],
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="x"),
                            showlegend=False
                        ))
                        
                        # Draw vertical reference line
                        fig2.add_shape(
                            type="line",
                            x0=case_time, x1=case_time,
                            y0=50, y1=y_case,
                            line=dict(color="red", width=1, dash="dash")
                        )
                        
                        # Draw controls (4 per case)
                        for j in range(4):
                            fig2.add_trace(go.Scatter(
                                x=[case_time],
                                y=[y_case + 10 + j*5],
                                mode="markers",
                                marker=dict(color="blue", size=8),
                                showlegend=False
                            ))
                    
                    fig2.update_layout(
                        height=250,
                        showlegend=False,
                        title="Nested Case-Control",
                        xaxis=dict(title="Time (years)"),
                        yaxis=dict(showticklabels=False)
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col3:
                    st.write("### Case-Cohort")
                    
                    # Create metrics display
                    st.metric("Sample Size", n_cc_sample, delta=f"{-round((1-n_cc_sample/n_cohort)*100)}%")
                    st.metric("Data Collection Cost", "Medium", delta="Lower")
                    st.metric("Statistical Efficiency", "~90%", delta="-10%")
                    
                    # Create simplified visualization
                    fig3 = go.Figure()
                    
                    # Draw timeline
                    fig3.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=50, y1=50,
                        line=dict(color="black", width=2)
                    )
                    
                    # Draw subcohort (random 20%)
                    subcohort_y = np.linspace(60, 140, 10)
                    for y in subcohort_y:
                        subject_time = np.random.uniform(1, followup_years)
                        is_case = np.random.random() < 0.2
                        
                        fig3.add_shape(
                            type="line",
                            x0=0, x1=subject_time,
                            y0=y, y1=y,
                            line=dict(color="green", width=2)
                        )
                        
                        if is_case:
                            fig3.add_trace(go.Scatter(
                                x=[subject_time],
                                y=[y],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                showlegend=False
                            ))
                    
                    # Draw non-subcohort cases
                    for i in range(5):
                        case_time = np.random.uniform(1, followup_years)
                        y_case = 160 + i*10
                        
                        fig3.add_shape(
                            type="line",
                            x0=0, x1=case_time,
                            y0=y_case, y1=y_case,
                            line=dict(color="gray", width=1, dash="dot")
                        )
                        
                        fig3.add_trace(go.Scatter(
                            x=[case_time],
                            y=[y_case],
                            mode="markers",
                            marker=dict(color="red", size=8, symbol="x"),
                            showlegend=False
                        ))
                    
                    fig3.update_layout(
                        height=250,
                        showlegend=False,
                        title="Case-Cohort Design",
                        xaxis=dict(title="Time (years)"),
                        yaxis=dict(showticklabels=False)
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Add a summary comparison table
                st.subheader("Design Comparison Summary")
                
                comparison_df = pd.DataFrame({
                    'Design': ['Full Cohort', 'Nested Case-Control', 'Case-Cohort'],
                    'Sample Size': [n_cohort, n_ncc_sample, n_cc_sample],
                    'Relative Cost': ['High', 'Medium', 'Medium'],
                    'Statistical Power': ['Excellent', 'Very Good', 'Good'],
                    'Multiple Outcomes': ['Yes', 'No', 'Yes'],
                    'Time-varying Exposures': ['Yes', 'Yes', 'Limited'],
                    'Key Advantage': [
                        'Complete follow-up data', 
                        'Efficient for single outcome', 
                        'Efficient for multiple outcomes'
                    ]
                })
                
                st.dataframe(comparison_df)
                
                # Add a scenario-based recommendation system
                st.subheader("Which Design Should I Use?")
                
                scenario = st.selectbox(
                    "Select your research scenario:",
                    [
                        "Single rare disease with expensive biomarker",
                        "Multiple related diseases in the same cohort",
                        "Common disease with time-varying exposure",
                        "Need to estimate absolute risks and population rates",
                        "Limited budget with expensive exposure assessment"
                    ],
                    key="ncc_scenario_selector"
                )
                
                recommendations = {
                    "Single rare disease with expensive biomarker": {
                        "Recommended": "Nested Case-Control",
                        "Explanation": "Nested case-control is ideal for rare diseases where biomarker assessment is expensive. It maintains statistical efficiency while dramatically reducing costs."
                    },
                    "Multiple related diseases in the same cohort": {
                        "Recommended": "Case-Cohort",
                        "Explanation": "Case-cohort design allows you to study multiple outcomes using the same subcohort, reducing redundant control selection."
                    },
                    "Common disease with time-varying exposure": {
                        "Recommended": "Full Cohort",
                        "Explanation": "For complex time-varying exposures, a full cohort analysis provides the most flexibility and avoids potential selection biases."
                    },
                    "Need to estimate absolute risks and population rates": {
                        "Recommended": "Case-Cohort",
                        "Explanation": "Case-cohort design allows estimation of absolute risks because the subcohort represents the full cohort. Nested case-control designs only provide relative measures."
                    },
                    "Limited budget with expensive exposure assessment": {
                        "Recommended": "Nested Case-Control or Case-Cohort",
                        "Explanation": "Both designs substantially reduce cost. Choose nested case-control for a single outcome or case-cohort for multiple outcomes."
                    }
                }
                
                if scenario in recommendations:
                    rec = recommendations[scenario]
                    st.success(f"**Recommended Design: {rec['Recommended']}**")
                    st.write(rec["Explanation"])
            
            # Run the comparison
            design_comparison()
    
    # ENHANCED CASE-COHORT STUDY SECTION
    elif study_design == "Case-Cohort Study":
        st.header("Case-Cohort Study Simulation")
        
        # Create tabs for different aspects of the case-cohort visualization
        tab_main, tab_multidisease, tab_interactive, tab_compare = st.tabs([
            "📉 Main Visualization", 
            "📝 Conceptual Overview", 
            "📊 Interactive Exploration", 
            "⚖️ Design Comparison"
        ])
        
        # Parameters for case-cohort study
        with tab_main:
            col1, col2 = st.columns(2)
            with col1:
                n_cohort = st.slider("Initial Cohort Size", 1000, 10000, 5000, key="cc_cohort_size")
                followup_years = st.slider("Follow-up Years", 1, 10, 5, key="cc_followup")
            with col2:
                subcohort_fraction = st.slider("Subcohort Sampling Fraction", 0.1, 0.5, 0.2, key="cc_fraction")
                exposure_effect = st.slider("Exposure Effect (Hazard Ratio)", 1.0, 5.0, 2.0, key="cc_effect")
        
        # Function to generate multi-outcome cohort data
        def generate_multi_outcome_data(n_cohort, followup_years):
            """Generate cohort data with multiple outcomes"""
            # Create baseline data
            data = pd.DataFrame({
                'id': range(n_cohort),
                'exposure': np.random.binomial(1, 0.3, n_cohort),
                'age': np.random.normal(50, 10, n_cohort),
                'sex': np.random.binomial(1, 0.5, n_cohort)
            })
            
            # Generate times to different events
            # Disease A - common disease
            baseline_hazard_A = 0.08
            lambda_A = baseline_hazard_A * np.exp(np.log(1.5) * data['exposure'])
            data['time_to_A'] = np.random.exponential(1/lambda_A)
            
            # Disease B - rare disease
            baseline_hazard_B = 0.03
            lambda_B = baseline_hazard_B * np.exp(np.log(2.0) * data['exposure'])
            data['time_to_B'] = np.random.exponential(1/lambda_B)
            
            # Disease C - another disease
            baseline_hazard_C = 0.05
            lambda_C = baseline_hazard_C * np.exp(np.log(1.2) * data['exposure'])
            data['time_to_C'] = np.random.exponential(1/lambda_C)
            
            # Censoring time
            data['time_to_censoring'] = np.random.uniform(0, followup_years, n_cohort)
            
            # Find earliest event and censoring
            data['earliest_event'] = data[['time_to_A', 'time_to_B', 'time_to_C']].min(axis=1)
            data['observed_time'] = np.minimum(data['earliest_event'], data['time_to_censoring'])
            
            # Determine which event occurred first (if any)
            data['event_type'] = None
            event_cols = ['time_to_A', 'time_to_B', 'time_to_C']
            event_types = ['Disease A', 'Disease B', 'Disease C']
            
            for col, event_type in zip(event_cols, event_types):
                mask = (data[col] == data['earliest_event']) & (data[col] <= data['time_to_censoring']) & (data[col] <= followup_years)
                data.loc[mask, 'event_type'] = event_type
            
            # Create indicator variables for each disease
            data['disease_A'] = (data['event_type'] == 'Disease A')
            data['disease_B'] = (data['event_type'] == 'Disease B')
            data['disease_C'] = (data['event_type'] == 'Disease C')
            
            return data
        
        # Generate data once to be used in all tabs
        np.random.seed(42)
        cohort_data = generate_multi_outcome_data(n_cohort, followup_years)
        
        # Select subcohort
        n_subcohort = int(n_cohort * subcohort_fraction)
        subcohort_ids = np.random.choice(cohort_data['id'].values, n_subcohort, replace=False)
        cohort_data['in_subcohort'] = cohort_data['id'].isin(subcohort_ids)
        
        # Main visualization tab
        with tab_main:
            # Step-by-step progression slider
            steps = st.radio(
                "Study Design Stages:",
                ["1. Initial Cohort", "2. Subcohort Selection", "3. Follow-up Period", 
                "4. Case Identification", "5. Analysis"],
                horizontal=True,
                key="cc_steps"
            )
            
            # Create dynamic visualization based on selected step
            fig = go.Figure()
            
            # Constants for visualization
            total_width = 800
            total_height = 600
            timeline_y = 350
            
            # Draw timeline
            fig.add_shape(
                type="line",
                x0=50, x1=750,
                y0=timeline_y, y1=timeline_y,
                line=dict(color="black", width=3),
            )
            
            # Add time labels
            for year in range(followup_years + 1):
                fig.add_annotation(
                    x=50 + (year * 700 / followup_years),
                    y=timeline_y - 20,
                    text=f"Year {year}",
                    showarrow=False,
                    font=dict(size=10)
                )
            
            # Different elements based on step
            if steps == "1. Initial Cohort":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=20, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=650,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=400, y=timeline_y - 100,
                    text="Step 1: Define and recruit a cohort of individuals<br>" +
                        f"• Cohort size: {n_cohort} participants<br>" +
                        "• Collect baseline information on all participants<br>" +
                        "• Store biospecimens for future analysis",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "2. Subcohort Selection":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=20, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw subcohort selection
                fig.add_shape(
                    type="rect",
                    x0=10, x1=90,
                    y0=timeline_y - 120, y1=timeline_y - 40,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y - 80,
                    text=f"Random Subcohort<br>n={n_subcohort}<br>({subcohort_fraction:.0%} of cohort)",
                    showarrow=False,
                    font=dict(size=12)
                )
                

                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=650,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=400, y=timeline_y - 100,
                    text="Step 2: Select a random subcohort<br>" +
                        f"• Subcohort fraction: {subcohort_fraction:.0%}<br>" +
                        f"• Subcohort size: {n_subcohort} participants<br>" +
                        "• Selection is done at baseline, before follow-up",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "3. Follow-up Period":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=20, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw subcohort selection
                fig.add_shape(
                    type="rect",
                    x0=10, x1=90,
                    y0=timeline_y - 150, y1=timeline_y - 70,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y - 110,
                    text=f"Random Subcohort<br>n={n_subcohort}",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Draw sample of subjects being followed
                # Subcohort subjects
                n_sample = 10  # Number of subjects to show
                sample_data_subcohort = cohort_data[cohort_data['in_subcohort']].sample(min(n_sample, n_subcohort))
                
                for i, (_, subject) in enumerate(sample_data_subcohort.iterrows()):
                    y_pos = timeline_y + 40 - i * 10
                    x_end = 50 + (subject['observed_time'] / followup_years) * 700
                    
                    # Subject's follow-up line
                    fig.add_shape(
                        type="line",
                        x0=50, x1=x_end,
                        y0=y_pos, y1=y_pos,
                        line=dict(
                            color="green",
                            width=2,
                            dash="solid"
                        ),
                    )
                    
                    # Add event marker if subject becomes a case
                    if subject['event_type'] is not None:
                        fig.add_trace(go.Scatter(
                            x=[x_end],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="x"),
                            name="Case",
                            showlegend=False
                        ))
                
                # Non-subcohort subjects
                sample_data_nonsubcohort = cohort_data[~cohort_data['in_subcohort']].sample(min(n_sample, n_cohort - n_subcohort))
                
                for i, (_, subject) in enumerate(sample_data_nonsubcohort.iterrows()):
                    y_pos = timeline_y - 150 - i * 10
                    x_end = 50 + (subject['observed_time'] / followup_years) * 700
                    
                    # Subject's follow-up line (lighter for non-subcohort)
                    fig.add_shape(
                        type="line",
                        x0=50, x1=x_end,
                        y0=y_pos, y1=y_pos,
                        line=dict(
                            color="gray",
                            width=1.5,
                            dash="dot"
                        ),
                    )
                    
                    # Add event marker if subject becomes a case
                    if subject['event_type'] is not None:
                        fig.add_trace(go.Scatter(
                            x=[x_end],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="x"),
                            name="Case (Non-subcohort)",
                            showlegend=False
                        ))
                
                # Add explanation
                fig.add_shape(
                    type="rect",
                    x0=150, x1=650,
                    y0=timeline_y - 150, y1=timeline_y - 50,
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=400, y=timeline_y - 100,
                    text="Step 3: Follow entire cohort over time<br>" +
                        f"• Follow-up period: {followup_years} years<br>" +
                        "• Monitor for disease outcomes<br>" +
                        "• Track multiple diseases simultaneously",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
            elif steps == "4. Case Identification":
                # Draw cohort box at baseline
                fig.add_shape(
                    type="rect",
                    x0=10, x1=80,
                    y0=timeline_y + 50, y1=timeline_y + 150,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(200, 200, 200, 0.3)"
                )
                
                fig.add_annotation(
                    x=45, y=timeline_y + 100,
                    text=f"Initial Cohort<br>n={n_cohort}",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                # Draw subcohort selection
                fig.add_shape(
                    type="rect",
                    x0=10, x1=90,
                    y0=timeline_y - 120, y1=timeline_y - 40,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=50, y=timeline_y - 80,
                    text=f"Random Subcohort<br>n={n_subcohort}",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Count cases of each type
                n_disease_A = sum(cohort_data['disease_A'])
                n_disease_B = sum(cohort_data['disease_B'])
                n_disease_C = sum(cohort_data['disease_C'])
                
                # Move the Step 4 explanation box below the subcohort box
                fig.add_shape(
                    type="rect",
                    x0=10, x1=210,  # Positioned under subcohort box
                    y0=timeline_y - 250, y1=timeline_y - 150,  # Below subcohort box
                    line=dict(color="black", width=1),
                    fillcolor="white"
                )
                
                fig.add_annotation(
                    x=110, y=timeline_y - 200,  # Center of the new box position
                    text="Step 4: Identify all cases<br>" +
                        f"• Disease A: {n_disease_A} cases<br>" +
                        f"• Disease B: {n_disease_B} cases<br>" +
                        f"• Disease C: {n_disease_C} cases<br>",
                    showarrow=False,
                    font=dict(size=14),
                    align="left"
                )
                
                # Draw case identification - moved to right side
                case_types = ['Disease A', 'Disease B', 'Disease C']
                case_counts = [n_disease_A, n_disease_B, n_disease_C]
                case_colors = ['red', 'purple', 'orange']
                
                # Position the disease boxes on the right side where the explanation was
                for i, (disease, count, color) in enumerate(zip(case_types, case_counts, case_colors)):
                    x_pos = 650  # Center position for right side boxes
                    y_pos = timeline_y - 120 - i * 70  # Vertical spacing between boxes
                    
                    # Box for disease cases
                    fig.add_shape(
                        type="rect",
                        x0=x_pos - 70, x1=x_pos + 70,
                        y0=y_pos - 25, y1=y_pos + 25,
                        line=dict(color=color, width=2),
                        fillcolor=f"rgba({255 if color=='red' else (128 if color=='purple' else 255)}, {0 if color=='red' else (0 if color=='purple' else 165)}, {0 if color=='red' else (128 if color=='purple' else 0)}, 0.2)"
                    )
                    
                    fig.add_annotation(
                        x=x_pos, y=y_pos,
                        text=f"{disease} Cases<br>n={count}",
                        showarrow=False,
                        font=dict(size=12)
                    )
                    
                    # Arrow from timeline to cases - adjust endpoint to match new box positions
                    fig.add_annotation(
                        x=200, y=timeline_y,
                        ax=x_pos - 70, ay=y_pos,  # Point to left edge of disease boxes
                        text="",
                        showarrow=True,
                        arrowhead=2,
                        arrowwidth=1.5,
                        arrowcolor=color
                    )                
            elif steps == "5. Analysis":
                # Get counts for each group
                n_subcohort_cases_A = sum(cohort_data['in_subcohort'] & cohort_data['disease_A'])
                n_subcohort_cases_B = sum(cohort_data['in_subcohort'] & cohort_data['disease_B'])
                n_subcohort_cases_C = sum(cohort_data['in_subcohort'] & cohort_data['disease_C'])
                
                n_nonsubcohort_cases_A = sum(~cohort_data['in_subcohort'] & cohort_data['disease_A'])
                n_nonsubcohort_cases_B = sum(~cohort_data['in_subcohort'] & cohort_data['disease_B'])
                n_nonsubcohort_cases_C = sum(~cohort_data['in_subcohort'] & cohort_data['disease_C'])
                
                n_subcohort_noncases = n_subcohort - n_subcohort_cases_A - n_subcohort_cases_B - n_subcohort_cases_C
                
                # Draw analysis diagram
                # Subcohort box
                fig.add_shape(
                    type="rect",
                    x0=200, x1=350,
                    y0=timeline_y - 50, y1=timeline_y + 50,
                    line=dict(color="green", width=2),
                    fillcolor="rgba(0, 128, 0, 0.2)"
                )
                
                fig.add_annotation(
                    x=275, y=timeline_y,
                    text=f"Subcohort<br>n={n_subcohort}",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # External cases box
                fig.add_shape(
                    type="rect",
                    x0=450, x1=600,
                    y0=timeline_y - 50, y1=timeline_y + 50,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255, 0, 0, 0.2)"
                )
                
                n_external_cases = n_nonsubcohort_cases_A + n_nonsubcohort_cases_B + n_nonsubcohort_cases_C
                
                fig.add_annotation(
                    x=525, y=timeline_y,
                    text=f"External Cases<br>n={n_external_cases}",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Analysis sample box
                fig.add_shape(
                    type="rect",
                    x0=325, x1=475,
                    y0=timeline_y - 150, y1=timeline_y - 100,
                    line=dict(color="purple", width=2),
                    fillcolor="rgba(128, 0, 128, 0.2)"
                )
                
                n_analysis_sample = n_subcohort + n_external_cases
                
                fig.add_annotation(
                    x=400, y=timeline_y - 125,
                    text=f"Analysis Sample<br>n={n_analysis_sample}",
                    showarrow=False,
                    font=dict(size=12)
                )
            
                
                fig.add_annotation(
                    x=400, y=timeline_y - 212.5,
                    text="Step 5: Analyze subcohort + all cases<br>" +
                        f"• Total sample size: {n_analysis_sample} ({n_analysis_sample/n_cohort:.1%} of cohort)<br>" +
                        "• Weighting required in analysis (to account for oversampling of cases)<br>" +
                        "• Can study multiple diseases with the same subcohort",
                    showarrow=False,
                    font=dict(size=14),
                    align="left",
                    bordercolor="black", 
                    borderwidth=1,
                    borderpad=8
                )
            
            # Add legend
            legend_items = []
            if steps in ["3. Follow-up Period", "4. Case Identification"]:
                legend_items.extend([
                    {"name": "Subcohort", "color": "green", "dash": "solid", "type": "line"},
                    {"name": "Non-subcohort", "color": "gray", "dash": "dot", "type": "line"},
                    {"name": "Case occurrence", "color": "red", "symbol": "x", "type": "marker"}
                ])
            elif steps == "5. Analysis":
                legend_items.extend([
                    {"name": "Subcohort", "color": "green", "type": "box"},
                    {"name": "External Cases", "color": "red", "type": "box"},
                    {"name": "Analysis Sample", "color": "purple", "type": "box"}
                ])
            
            # Add legend outside the chart
            if legend_items:
                legend_x = 750
                legend_y_start = timeline_y + 100
                
                for i, item in enumerate(legend_items):
                    y_pos = legend_y_start - i * 30
                    
                    if item["type"] == "line":
                        fig.add_shape(
                            type="line",
                            x0=legend_x - 30, x1=legend_x,
                            y0=y_pos, y1=y_pos,
                            line=dict(
                                color=item["color"],
                                width=2,
                                dash=item.get("dash", "solid")
                            ),
                        )
                    elif item["type"] == "marker":
                        fig.add_trace(go.Scatter(
                            x=[legend_x - 15],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(
                                color=item["color"],
                                size=10,
                                symbol=item.get("symbol", "circle")
                            ),
                            showlegend=False
                        ))
                    elif item["type"] == "box":
                        fig.add_shape(
                            type="rect",
                            x0=legend_x - 30, x1=legend_x,
                            y0=y_pos - 10, y1=y_pos + 10,
                            line=dict(color=item["color"], width=2),
                            fillcolor=f"rgba({255 if item['color']=='red' else (0 if item['color']=='green' else 128)}, {0 if item['color']=='red' else (128 if item['color']=='green' else 0)}, {0 if item['color']=='red' else (0 if item['color']=='green' else 128)}, 0.2)"
                        )
                    
                    fig.add_annotation(
                        x=legend_x + 10,
                        y=y_pos,
                        text=item["name"],
                        showarrow=False,
                        font=dict(size=12),
                        xanchor="left"
                    )
            
            # Update layout
            fig.update_layout(
                title="Case-Cohort Study Design",
                height=total_height,
                plot_bgcolor="white",
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[0, 850]  # Include space for legend
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[timeline_y - 300, timeline_y + 200]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Multiple disease advantage tab
        with tab_multidisease:
            st.subheader("Case-Cohort Design: Multiple Disease Advantage")
            
            st.write("""
            One key advantage of the case-cohort design is the ability to study multiple disease outcomes 
            using the same subcohort. This is more efficient than nested case-control studies when 
            multiple outcomes are of interest.
            """)
            
            # Calculate metrics for each disease
            disease_metrics = []
            for disease, letter in zip(['disease_A', 'disease_B', 'disease_C'], ['A', 'B', 'C']):
                # Get cases
                cases = cohort_data[cohort_data[disease]]
                n_cases = len(cases)
                
                # Subcohort cases
                subcohort_cases = cases[cases['in_subcohort']]
                n_subcohort_cases = len(subcohort_cases)
                
                # External cases
                n_external_cases = n_cases - n_subcohort_cases
                
                # Analysis set size
                analysis_set_size = n_subcohort + n_external_cases
                
                # NCC comparison (4 controls per case)
                ncc_size = n_cases * 5  # 1 case + 4 controls
                
                disease_metrics.append({
                    'Disease': f'Disease {letter}',
                    'Total Cases': n_cases,
                    'Subcohort Cases': n_subcohort_cases,
                    'External Cases': n_external_cases,
                    'Analysis Set Size': analysis_set_size,
                    'NCC Size': ncc_size
                })
            
            # Create comparison table
            metrics_df = pd.DataFrame(disease_metrics)
            st.write("### Sample Size Requirements by Study Design")
            st.dataframe(metrics_df)
            
            # Visualize with grouped bar chart for each disease
            plot_data = []
            for disease in disease_metrics:
                plot_data.append({
                    'Disease': disease['Disease'],
                    'Method': 'Case-Cohort',
                    'Sample Size': disease['Analysis Set Size']
                })
                plot_data.append({
                    'Disease': disease['Disease'],
                    'Method': 'Nested Case-Control',
                    'Sample Size': disease['NCC Size']
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create the bar chart
            fig = px.bar(
                plot_df,
                x='Disease',
                y='Sample Size',
                color='Method',
                barmode='group',
                title='Required Sample Size: Case-Cohort vs. Nested Case-Control',
                color_discrete_map={
                    'Case-Cohort': '#1f77b4',
                    'Nested Case-Control': '#ff7f0e'
                }
            )
            
            # Calculate total sample size for all diseases
            cc_total = sum(d['Analysis Set Size'] for d in disease_metrics)
            ncc_total = sum(d['NCC Size'] for d in disease_metrics)
            
            # Add annotation for total sample size
            fig.add_annotation(
                x=1.5,  # Center of the plot
                y=max(plot_df['Sample Size']) * 1.3,
                text=f"Total Sample Size:<br>Case-Cohort: {cc_total}<br>Nested Case-Control: {ncc_total}",
                showarrow=False,
                font=dict(size=12),
                bordercolor="black",
                borderwidth=1,
                borderpad=8,
                bgcolor="white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Side-by-side comparison
            st.write("### Sampling Strategy Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Nested Case-Control")
                st.write("""
                Must select separate controls for each disease outcome:
                - Disease A: 1 case + 4 controls
                - Disease B: 1 case + 4 controls 
                - Disease C: 1 case + 4 controls
                """)
                
                # Simple visualization
                ncc_fig = go.Figure()
                
                # Draw the three nested case-control sets
                y_positions = [1, 3, 5]
                for i, y in enumerate(y_positions):
                    # Add case
                    ncc_fig.add_trace(go.Scatter(
                        x=[0],
                        y=[y],
                        mode="markers",
                        marker=dict(color="red", size=12, symbol="star"),
                        name=f"Disease {chr(65 + i)} Case" if i == 0 else None,
                        showlegend=i == 0
                    ))
                    
                    # Add controls
                    for j in range(4):
                        ncc_fig.add_trace(go.Scatter(
                            x=[j + 1],
                            y=[y],
                            mode="markers",
                            marker=dict(color="blue", size=10),
                            name="Control" if i == 0 and j == 0 else None,
                            showlegend=i == 0 and j == 0
                        ))
                    
                    # Add disease label
                    ncc_fig.add_annotation(
                        x=-1,
                        y=y,
                        text=f"Disease {chr(65 + i)}",
                        showarrow=False,
                        xanchor="right"
                    )
                
                # Update layout
                ncc_fig.update_layout(
                    title="Nested Case-Control Design",
                    height=400,
                    xaxis=dict(
                        title="Case and Controls",
                        showticklabels=False,
                        range=[-2, 5]
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        range=[0, 6]
                    )
                )
                
                st.plotly_chart(ncc_fig, use_container_width=True)
            
            with col2:
                st.write("#### Case-Cohort")
                st.write("""
                One randomly selected subcohort is used for all disease outcomes:
                - Same subcohort for diseases A, B, and C
                - Only add cases not already in subcohort
                - Reuse exposure measurements
                """)
                
                # Simple visualization for case-cohort
                cc_fig = go.Figure()
                
                # Draw the subcohort
                for i in range(10):
                    cc_fig.add_trace(go.Scatter(
                        x=[0],
                        y=[i],
                        mode="markers",
                        marker=dict(color="green", size=10),
                        name="Subcohort Member" if i == 0 else None,
                        showlegend=i == 0
                    ))
                
                # Add external cases for each disease
                for i, (y, disease) in enumerate(zip([2, 5, 8], ['A', 'B', 'C'])):
                    cc_fig.add_trace(go.Scatter(
                        x=[2],
                        y=[y],
                        mode="markers",
                        marker=dict(color="red", size=12, symbol="star"),
                        name=f"Disease {disease} Case (not in subcohort)" if i == 0 else None,
                        showlegend=i == 0
                    ))
                    
                    # Add arrow connecting to subcohort
                    cc_fig.add_annotation(
                        x=1,
                        y=y,
                        ax=0.5,
                        ay=y,
                        text="",
                        showarrow=True,
                        arrowhead=3,
                        arrowwidth=1.5
                    )
                    
                    # Add disease label
                    cc_fig.add_annotation(
                        x=3,
                        y=y,
                        text=f"Disease {disease}",
                        showarrow=False,
                        xanchor="left"
                    )
                
                # Add labels
                cc_fig.add_annotation(
                    x=0,
                    y=11,
                    text="Random Subcohort",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                cc_fig.add_annotation(
                    x=2,
                    y=11,
                    text="Cases Outside Subcohort",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Update layout
                cc_fig.update_layout(
                    title="Case-Cohort Design",
                    height=400,
                    xaxis=dict(
                        showticklabels=False,
                        range=[-1, 4]
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        range=[-1, 12]
                    )
                )
                
                st.plotly_chart(cc_fig, use_container_width=True)
            
            # Efficiency explanation
            st.write("""
            ### Efficiency Advantage
            
            For a single disease, nested case-control and case-cohort designs have comparable efficiency. However, 
            when studying multiple diseases from the same cohort, the case-cohort design becomes more efficient:
            
            - In a nested case-control study, control selection must be repeated for each disease
            - In a case-cohort study, the same subcohort serves as the comparison group for all diseases
            - Biomarker assessment or exposure measurement is done once in the subcohort
            - The efficiency advantage increases with the number of outcomes studied
            """)
        
        # Interactive tab
        with tab_interactive:
            st.subheader("Interactive Subcohort Selection")
            
            # Allow user to adjust subcohort fraction
            adjusted_fraction = st.slider(
                "Adjust subcohort fraction:", 
                min_value=0.1, 
                max_value=0.5, 
                value=subcohort_fraction,
                step=0.05,
                key="cc_adjusted_fraction"
            )
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            # Calculate various metrics for the adjusted fraction
            adj_n_subcohort = int(n_cohort * adjusted_fraction)
            
            # Count cases of each type
            n_disease_A = sum(cohort_data['disease_A'])
            n_disease_B = sum(cohort_data['disease_B'])
            n_disease_C = sum(cohort_data['disease_C'])
            
            # Expected cases in subcohort
            exp_subcohort_A = n_disease_A * adjusted_fraction
            exp_subcohort_B = n_disease_B * adjusted_fraction
            exp_subcohort_C = n_disease_C * adjusted_fraction
            
            # Expected external cases
            exp_external_A = n_disease_A - exp_subcohort_A
            exp_external_B = n_disease_B - exp_subcohort_B
            exp_external_C = n_disease_C - exp_subcohort_C
            
            # Total sample size
            exp_sample_size = adj_n_subcohort + exp_external_A + exp_external_B + exp_external_C
            
            # Display metrics
            with col1:
                st.metric("Subcohort Size", f"{adj_n_subcohort} ({adjusted_fraction:.0%} of cohort)")
                st.metric("Total Sample Size for Analysis", int(exp_sample_size))
                st.metric("Percentage of Original Cohort", f"{exp_sample_size/n_cohort:.1%}")
                
                # Calculate cost savings
                # Assuming $100 per subject for exposure assessment
                full_cohort_cost = n_cohort * 100
                case_cohort_cost = exp_sample_size * 100
                cost_savings = full_cohort_cost - case_cohort_cost
                
                st.metric("Cost Savings", f"${cost_savings:,.0f}")
            
            with col2:
                # Create a plot showing efficiency metrics
                # Calculate for a range of fractions
                fractions = np.arange(0.1, 0.55, 0.05)
                relative_efficiencies = []
                sample_sizes = []
                
                for fraction in fractions:
                    # Sample size calculation
                    subcohort = int(n_cohort * fraction)
                    cases_in_subcohort = n_disease_A * fraction + n_disease_B * fraction + n_disease_C * fraction
                    external_cases = (n_disease_A + n_disease_B + n_disease_C) - cases_in_subcohort
                    
                    sample_size = subcohort + external_cases
                    sample_sizes.append(sample_size)
                    
                    # Efficiency calculation (simplified model)
                    # Efficiency increases with subcohort size but has diminishing returns
                    rel_efficiency = np.sqrt(fraction)
                    relative_efficiencies.append(rel_efficiency)
                
                # Create dataframe for plotting
                efficiency_df = pd.DataFrame({
                    'Subcohort Fraction': fractions,
                    'Relative Statistical Efficiency': relative_efficiencies,
                    'Sample Size': sample_sizes
                })
                
                # Plot statistical efficiency
                fig1 = px.line(
                    efficiency_df,
                    x='Subcohort Fraction',
                    y='Relative Statistical Efficiency',
                    title='Statistical Efficiency vs. Subcohort Size'
                )
                
                # Highlight selected fraction
                fig1.add_shape(
                    type="line",
                    x0=adjusted_fraction, x1=adjusted_fraction,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Update layout
                fig1.update_layout(
                    xaxis=dict(tickformat=".0%"),
                    yaxis=dict(
                        title="Relative Efficiency",
                        range=[0, 1.1]
                    )
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Plot sample size
                fig2 = px.line(
                    efficiency_df,
                    x='Subcohort Fraction',
                    y='Sample Size',
                    title='Required Sample Size vs. Subcohort Fraction'
                )
                
                # Highlight selected fraction
                fig2.add_shape(
                    type="line",
                    x0=adjusted_fraction, x1=adjusted_fraction,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Update layout
                fig2.update_layout(
                    xaxis=dict(tickformat=".0%")
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Explain the tradeoff
            st.write("""
            ### Understanding the Tradeoff
            
            As you increase the subcohort fraction:
            
            - **Statistical efficiency increases** - larger subcohort provides better precision
            - **Sample size increases** - more subjects need exposure assessment
            - **Cost increases** - more resources required
            
            The optimal subcohort size balances statistical efficiency with resource constraints.
            Typically, subcohort fractions of 10-20% provide good efficiency while minimizing costs.
            """)
            
            # Subcohort sampling simulator
            st.subheader("Subcohort Sampling Simulator")
            
            if st.button("Simulate different subcohort samples", key="cc_sim_button"):
                # Create multiple subcohort samples to show variability
                n_simulations = 3
                sim_results = []
                
                for i in range(n_simulations):
                    # Select random subcohort
                    sim_subcohort_ids = np.random.choice(
                        cohort_data['id'].values, 
                        adj_n_subcohort, 
                        replace=False
                    )
                    
                    # Count cases in subcohort
                    subcohort_mask = cohort_data['id'].isin(sim_subcohort_ids)
                    sub_cases_A = sum(subcohort_mask & cohort_data['disease_A'])
                    sub_cases_B = sum(subcohort_mask & cohort_data['disease_B'])
                    sub_cases_C = sum(subcohort_mask & cohort_data['disease_C'])
                    
                    # Calculate external cases
                    ext_cases_A = n_disease_A - sub_cases_A
                    ext_cases_B = n_disease_B - sub_cases_B
                    ext_cases_C = n_disease_C - sub_cases_C
                    
                    # Total sample size
                    total_sample = adj_n_subcohort + ext_cases_A + ext_cases_B + ext_cases_C
                    
                    sim_results.append({
                        'Simulation': i+1,
                        'Subcohort Cases A': sub_cases_A,
                        'Subcohort Cases B': sub_cases_B,
                        'Subcohort Cases C': sub_cases_C,
                        'External Cases A': ext_cases_A,
                        'External Cases B': ext_cases_B,
                        'External Cases C': ext_cases_C,
                        'Total Sample Size': total_sample
                    })
                
                # Display results
                sim_df = pd.DataFrame(sim_results)
                st.dataframe(sim_df)
                
                st.write("""
                **Note how random variation in subcohort selection affects:**
                1. The number of cases captured in the subcohort
                2. The number of external cases that need to be added
                3. The total analysis sample size
                
                Despite this variation, the expected sample size remains relatively stable.
                """)
        
        # Design comparison tab
        with tab_compare:
            st.subheader("Study Design Comparison: Case-Cohort vs. Other Designs")
            
            # Reuse the design comparison function from nested case-control with minor modifications
            def cc_design_comparison():
                # Create a 3-column layout
                col1, col2, col3 = st.columns(3)
                
                # Calculate key metrics
                n_cases_total = sum(cohort_data['disease_A'] | cohort_data['disease_B'] | cohort_data['disease_C'])
                n_controls_per_case = 4
                n_ncc_sample = n_cases_total * (1 + n_controls_per_case)
                n_subcohort = int(n_cohort * subcohort_fraction)
                n_cases_not_in_subcohort = sum(
                    ~cohort_data['in_subcohort'] & 
                    (cohort_data['disease_A'] | cohort_data['disease_B'] | cohort_data['disease_C'])
                )
                n_cc_sample = n_subcohort + n_cases_not_in_subcohort
                
                with col1:
                    st.write("### Full Cohort")
                    
                    # Create metrics display
                    st.metric("Sample Size", n_cohort)
                    st.metric("Data Collection Cost", "High", delta="Baseline")
                    st.metric("Statistical Efficiency", "100%", delta="Baseline")
                    
                    # Same visualization as in nested case-control
                    fig1 = go.Figure()
                    
                    # Draw timeline
                    fig1.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=50, y1=50,
                        line=dict(color="black", width=2)
                    )
                    
                    # Draw subjects (simplified)
                    y_positions = np.linspace(60, 200, 20)
                    
                    for i, y in enumerate(y_positions):
                        # Draw follow-up line
                        subject_time = np.random.uniform(1, followup_years)
                        is_case = np.random.random() < 0.2
                        
                        fig1.add_shape(
                            type="line",
                            x0=0, x1=subject_time,
                            y0=y, y1=y,
                            line=dict(color="gray", width=1.5)
                        )
                        
                        # Add case marker if applicable
                        if is_case:
                            fig1.add_trace(go.Scatter(
                                x=[subject_time],
                                y=[y],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                showlegend=False
                            ))
                    
                    fig1.update_layout(
                        height=400,
                        showlegend=False,
                        title="Full Cohort Follow-up",
                        xaxis=dict(title="Time (years)"),
                        yaxis=dict(showticklabels=False)
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.write("### Nested Case-Control")
                    
                    # Create metrics display
                    st.metric("Sample Size", n_ncc_sample, delta=f"{-round((1-n_ncc_sample/n_cohort)*100)}%")
                    st.metric("Data Collection Cost", "Medium", delta="Lower")
                    st.metric("Statistical Efficiency", "~95%", delta="-5%")
                    
                    # Same visualization as before
                    fig2 = go.Figure()
                    
                    # Draw timeline
                    fig2.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=50, y1=50,
                        line=dict(color="black", width=2)
                    )
                    
                    # Draw subjects (simplified with focus on case-control selection)
                    # Similar to fig1 but highlight control selection
                    case_times = [1.5, 3, 4.2]
                    
                    for i, case_time in enumerate(case_times):
                        y_case = 80 + i*30
                        
                        # Draw case
                        fig2.add_trace(go.Scatter(
                            x=[case_time],
                            y=[y_case],
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="x"),
                            showlegend=False
                        ))
                        
                        # Draw vertical reference line
                        fig2.add_shape(
                            type="line",
                            x0=case_time, x1=case_time,
                            y0=50, y1=y_case,
                            line=dict(color="red", width=1, dash="dash")
                        )
                        
                        # Draw controls (4 per case)
                        for j in range(4):
                            fig2.add_trace(go.Scatter(
                                x=[case_time],
                                y=[y_case + 10 + j*5],
                                mode="markers",
                                marker=dict(color="blue", size=8),
                                showlegend=False
                            ))
                    
                    fig2.update_layout(
                        height=250,
                        showlegend=False,
                        title="Nested Case-Control",
                        xaxis=dict(title="Time (years)"),
                        yaxis=dict(showticklabels=False)
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col3:
                    st.write("### Case-Cohort")
                    
                    # Create metrics display
                    st.metric("Sample Size", n_cc_sample, delta=f"{-round((1-n_cc_sample/n_cohort)*100)}%")
                    st.metric("Data Collection Cost", "Medium", delta="Lower")
                    st.metric("Statistical Efficiency", "~90%", delta="-10%")
                    
                    # Create visualization
                    fig3 = go.Figure()
                    
                    # Draw timeline
                    fig3.add_shape(
                        type="line",
                        x0=0, x1=followup_years,
                        y0=50, y1=50,
                        line=dict(color="black", width=2)
                    )
                    
                    # Draw subcohort (random 20%)
                    subcohort_y = np.linspace(60, 140, 10)
                    for y in subcohort_y:
                        subject_time = np.random.uniform(1, followup_years)
                        is_case = np.random.random() < 0.2
                        
                        fig3.add_shape(
                            type="line",
                            x0=0, x1=subject_time,
                            y0=y, y1=y,
                            line=dict(color="green", width=2)
                        )
                        
                        if is_case:
                            fig3.add_trace(go.Scatter(
                                x=[subject_time],
                                y=[y],
                                mode="markers",
                                marker=dict(color="red", size=8, symbol="x"),
                                showlegend=False
                            ))
                    
                    # Draw non-subcohort cases
                    for i in range(5):
                        case_time = np.random.uniform(1, followup_years)
                        y_case = 160 + i*10
                        
                        fig3.add_shape(
                            type="line",
                            x0=0, x1=case_time,
                            y0=y_case, y1=y_case,
                            line=dict(color="gray", width=1, dash="dot")
                        )
                        
                        fig3.add_trace(go.Scatter(
                            x=[case_time],
                            y=[y_case],
                            mode="markers",
                            marker=dict(color="red", size=8, symbol="x"),
                            showlegend=False
                        ))
                    
                    fig3.update_layout(
                        height=250,
                        showlegend=False,
                        title="Case-Cohort Design",
                        xaxis=dict(title="Time (years)"),
                        yaxis=dict(showticklabels=False)
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Add a summary comparison table
                st.subheader("Design Comparison Summary")
                
                comparison_df = pd.DataFrame({
                    'Design': ['Full Cohort', 'Nested Case-Control', 'Case-Cohort'],
                    'Sample Size': [n_cohort, n_ncc_sample, n_cc_sample],
                    'Relative Cost': ['High', 'Medium', 'Medium'],
                    'Statistical Power': ['Excellent', 'Very Good', 'Good'],
                    'Multiple Outcomes': ['Yes', 'No', 'Yes'],
                    'Time-varying Exposures': ['Yes', 'Yes', 'Limited'],
                    'Key Advantage': [
                        'Complete follow-up data', 
                        'Efficient for single outcome', 
                        'Efficient for multiple outcomes'
                    ]
                })
                
                st.dataframe(comparison_df)
                
                # Add a scenario-based recommendation system
                st.subheader("Which Design Should I Use?")
                
                scenario = st.selectbox(
                    "Select your research scenario:",
                    [
                        "Single rare disease with expensive biomarker",
                        "Multiple related diseases in the same cohort",
                        "Common disease with time-varying exposure",
                        "Need to estimate absolute risks and population rates",
                        "Limited budget with expensive exposure assessment"
                    ],
                    key="cc_scenario_selector"
                )
                
                recommendations = {
                    "Single rare disease with expensive biomarker": {
                        "Recommended": "Nested Case-Control",
                        "Explanation": "Nested case-control is ideal for rare diseases where biomarker assessment is expensive. It maintains statistical efficiency while dramatically reducing costs."
                    },
                    "Multiple related diseases in the same cohort": {
                        "Recommended": "Case-Cohort",
                        "Explanation": "Case-cohort design allows you to study multiple outcomes using the same subcohort, reducing redundant control selection."
                    },
                    "Common disease with time-varying exposure": {
                        "Recommended": "Full Cohort",
                        "Explanation": "For complex time-varying exposures, a full cohort analysis provides the most flexibility and avoids potential selection biases."
                    },
                    "Need to estimate absolute risks and population rates": {
                        "Recommended": "Case-Cohort",
                        "Explanation": "Case-cohort design allows estimation of absolute risks because the subcohort represents the full cohort. Nested case-control designs only provide relative measures."
                    },
                    "Limited budget with expensive exposure assessment": {
                        "Recommended": "Nested Case-Control or Case-Cohort",
                        "Explanation": "Both designs substantially reduce cost. Choose nested case-control for a single outcome or case-cohort for multiple outcomes."
                    }
                }
                
                if scenario in recommendations:
                    rec = recommendations[scenario]
                    st.success(f"**Recommended Design: {rec['Recommended']}**")
                    st.write(rec["Explanation"])
            
            # Run the comparison
            cc_design_comparison()
    
    # Educational content for each design
    st.header("Study Design Characteristics")

    if study_design == "Cohort Study":
        st.write("""
        **Key features of cohort studies:**
        1. Follow participants over time
        2. Compare exposed and unexposed groups
        3. Can measure multiple outcomes
        4. Can calculate incidence and relative risks
        5. May have issues with loss to follow-up
        """)
    elif study_design == "Case-Control Study":
        st.write("""
        **Key features of case-control studies:**
        1. Start with disease status (cases and controls)
        2. Look backwards at exposures
        3. Efficient for rare diseases
        4. Can only calculate odds ratios
        5. May have recall bias
        """)
    elif study_design == "Randomized Controlled Trial":
        st.write("""
        **Key features of RCTs:**
        1. Random allocation to intervention
        2. Can establish causality
        3. Balances confounders
        4. May have limited generalizability
        5. May have ethical constraints
        """)
        
    elif study_design == "Cross-sectional Study":
        st.write("""
        **Key features of Cross-sectional Studies:**
        1. **Data collection**
        - All data collected at a single point in time
        - Exposure and outcome measured simultaneously
        - No follow-up required
        - Typically uses surveys or existing data
        
        2. **Measures of Association**
        - Prevalence Ratio (PR): Direct comparison of disease prevalence between exposed and unexposed
        - Prevalence Odds Ratio (POR): Ratio of odds of disease in exposed vs. unexposed
        
        3. **Best Use Scenarios**
        - Health service planning and needs assessment
        - Disease surveillance and prevalence estimation
        - Generating hypotheses for future studies
        - Studying stable characteristics
        """)
        
    elif study_design == "Nested Case-Control Study":
        st.write("""
        **Key features of Nested Case-Control:**
        1. **Efficiency and Resources**
        - More cost-efficient than analyzing entire cohort
        - Reduces data collection costs while maintaining validity
        - Particularly useful for expensive or labor-intensive exposure measurements
        - Perfect for biomarker studies or when specimen analysis is costly
        
        2. **Temporal Aspects**
        - Preserves temporal relationship between exposure and outcome
        - Allows study of time-varying exposures
        - Can assess multiple time windows of exposure
        - Enables evaluation of exposure changes over time
        
        3. **Statistical Considerations**
        - Maintains the same validity as full cohort analysis
        - Allows matching on time-at-risk
        - Can control for confounding through matching
        - Power depends on number of controls per case
        
        4. **Best Use Scenarios**
        - Rare diseases in large cohorts
        - Studies requiring expensive biomarker analysis
        - When complete exposure assessment is impractical
        - When efficiency is needed without compromising validity
        """)
        
    elif study_design == "Case-Cohort Study":
        st.write("""
        **Key features of Case-Cohort:**
        
        1. **Design Features**
        - Random sample of the entire cohort (subcohort) plus all cases
        - No matching required
        - Can study multiple disease outcomes
        - Subcohort serves as a comparison group for all outcomes

        2. **Analytical Advantages**
        - Can estimate absolute risks and rates
        - Allows for time-varying covariates
        - Flexible analysis options
        - Can handle competing risks
        
        3. **Practical Benefits**
        - Exposure data needed only for subcohort and cases
        - Subcohort can be used for multiple outcomes
        - Can add new outcomes during study
        - More efficient than full cohort analysis
        
        4. **Best Use Scenarios**
        - Studies with multiple outcomes of interest
        - When exposure measurement is expensive
        - When population rates are needed
        - When matching is impractical or undesirable
        """)
        
        

    # Educational content for each design
    st.header("🧐 Test your understanding")

    if study_design == "Cohort Study":
        quiz_cohort = st.radio(
            "Which measure can be directly computed from a cohort study?",
            ("Select an answer", "Odds Ratio", "Risk Ratio", "Neither can be computed")
        )
        if quiz_cohort != "Select an answer":
            if quiz_cohort == "Risk Ratio":
                st.success("✅ Correct! Cohort studies allow you to calculate incidence and thus risk ratios.")
            else:
                st.error("❌ Not quite. While you can compute an odds ratio, the primary advantage of a cohort design is that you can directly compute a Risk Ratio.")


    elif study_design == "Case-Control Study":
        quiz_case_control = st.radio(
            "Which measure of association is primarily used in a case-control study?",
            ("Select an answer", "Odds Ratio (OR)", "Risk Ratio (RR)", "Risk Difference (RD)")
        )
        if quiz_case_control != "Select an answer":
            if quiz_case_control == "Odds Ratio (OR)":
                st.success("✅ Correct! Case-control designs typically estimate the odds ratio.")
            else:
                st.error("❌ In a case-control study, incidence cannot be directly computed, so the odds ratio is the primary measure.")
        

    elif study_design == "Randomized Controlled Trial":
        quiz_rct = st.radio(
            "Why are RCTs considered the gold standard in epidemiological research?",
            (   "Select an answer",
                "Because they are cheapest to conduct.",
                "They never lose participants to follow-up.",
                "They randomly assign exposure, minimizing confounding."
            )
        )
        if quiz_rct != "Select an answer":
            if quiz_rct == "They randomly assign exposure, minimizing confounding.":
                st.success("✅ Correct! Randomization helps ensure confounders are evenly distributed.")
            else:
                st.error("❌ Not quite. The key advantage is the random assignment of exposure.")

    elif study_design == "Cross-sectional Study":
        quiz_cross = st.radio(
            "What is a major limitation of cross-sectional studies for determining causality?",
            (
                "Select an answer",
                "They rely on randomization.",
                "They only capture data at one point in time, making temporality unclear.",
                "They require large budgets and are always unethical."
            )
        )
        if quiz_cross != "Select an answer":
            if quiz_cross == "They only capture data at one point in time, making temporality unclear.":
                st.success("✅ Exactly. In cross-sectional studies, exposure and outcome are measured simultaneously, so we can't confirm which came first.")
            else:
                st.error("❌ That's not the main reason. The biggest limitation is that we can't determine temporality with cross-sectional data.")

        
    elif study_design == "Nested Case-Control Study":
        quiz_nested_cc = st.radio(
            "Which of the following is a key advantage of a nested case-control study?",
            (
                "Select an answer",
                "It is more cost-efficient than assessing the entire cohort.",
                "You can easily compute incidence in the sub-sample.",
                "All participants are followed for the full duration, guaranteeing no loss to follow-up."
            )
        )
        if quiz_nested_cc != "Select an answer":
            if quiz_nested_cc == "It is more cost-efficient than assessing the entire cohort.":
                st.success("✅ Correct! Nested designs use fewer resources while maintaining validity.")
            else:
                st.error("❌ Not quite. The primary advantage is cost and resource efficiency while maintaining many cohort advantages.")

        
    elif study_design == "Case-Cohort Study":
        quiz_case_cohort = st.radio(
            "What is a main advantage of a case-cohort study design?",
            (
                "Select an answer",
                "It only requires cases, no controls.",
                "It does not require any follow-up period.",
                "You can investigate multiple outcomes using the same subcohort."
            )
        )
        if quiz_case_cohort != "Select an answer":
            if quiz_case_cohort == "You can investigate multiple outcomes using the same subcohort.":
                st.success("✅ Correct! The subcohort can be used to study various diseases that arise.")
            else:
                st.error("❌ Not quite. The hallmark advantage is the ability to examine multiple outcomes.")

with code_tab:
    study_designs_code.app()
