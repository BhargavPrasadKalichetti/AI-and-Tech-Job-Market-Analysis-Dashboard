# -----------------------------------------------------------
# Title: AI and Tech Job Market Analysis Dashboard
# Author: Bhargav Prasad Kalichetti
# Last Modified:  December 8, 2024
# Description: This Python script creates an interactive
#              dashboard using Streamlit to analyze trends
#              in AI, data science, and sales roles from
#              2010 to 2024. It includes dynamic filters,
#              KPIs, and visualizations such as line charts,
#              scatter plots, heatmaps, and more.
# -----------------------------------------------------------




import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set page config
st.set_page_config(page_title="Job Market Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Data Loading and Processing Functions
@st.cache_data
def load_and_clean_data():
    """Load and clean the dataset with proper error handling and data validation"""
    try:
        df = pd.read_csv('AIDataset.csv')
        
        # Handle missing values
        numeric_columns = ['Salary_USD', 'Job_Openings', 'Growth_Rate', 'Gender_Diversity_Index']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        # Convert datatypes
        df['Year'] = df['Year'].astype(int)
        df['Salary_USD'] = df['Salary_USD'].astype(float)
        df['Job_Openings'] = df['Job_Openings'].astype(int)
        df['Growth_Rate'] = df['Growth_Rate'].astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error in data loading and cleaning: {str(e)}")
        return None

# Visualization Functions
@st.cache_data
def create_salary_trend_chart(filtered_df):
    """Create an enhanced line chart for salary trends"""
    try:
        salary_trends = filtered_df.groupby(['Year', 'Job_Title'])['Salary_USD'].mean().reset_index()
        fig = px.line(salary_trends, 
                     x='Year', 
                     y='Salary_USD', 
                     color='Job_Title',
                     title='Salary Trends by Job Title (2010-2024)')
        fig.update_layout(
            hovermode='x unified',
            height=500,
            yaxis_title='Average Salary (USD)',
            xaxis_title='Year'
        )
        return fig
    except Exception as e:
        st.error(f"Error in salary trend chart: {str(e)}")
        return None

@st.cache_data
def create_salary_experience_scatter(filtered_df):
    """Create advanced scatter plot of salary vs experience"""
    try:
        # Calculate average salaries and job counts for each experience level and job title
        salary_exp_data = filtered_df.groupby(['Experience_Level', 'Job_Title']).agg({
            'Salary_USD': 'mean',
            'Job_Openings': 'sum'
        }).reset_index()

        fig = px.scatter(salary_exp_data,
                        x='Experience_Level',
                        y='Salary_USD',
                        color='Job_Title',
                        size='Job_Openings',
                        title='Salary Distribution by Experience Level and Job Title',
                        hover_data=['Job_Openings'])
        
        fig.update_layout(
            height=500,
            xaxis_title='Experience Level',
            yaxis_title='Average Salary (USD)',
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Error in salary-experience scatter plot: {str(e)}")
        return None

@st.cache_data
def create_salary_boxplot(filtered_df):
    """Create box plot for salary distribution analysis"""
    try:
        fig = px.box(filtered_df,
                    x='Job_Title',
                    y='Salary_USD',
                    color='Experience_Level',
                    title='Salary Distribution by Job Title and Experience Level')
        
        fig.update_layout(
            height=500,
            xaxis_title='Job Title',
            yaxis_title='Salary (USD)',
            xaxis_tickangle=-45
        )
        return fig
    except Exception as e:
        st.error(f"Error in salary box plot: {str(e)}")
        return None

@st.cache_data
def create_industry_job_distribution(filtered_df):
    """Create sunburst chart for industry and job distribution"""
    try:
        industry_job_data = filtered_df.groupby(['Industry', 'Job_Title']).agg({
            'Job_Openings': 'sum',
            'Salary_USD': 'mean'
        }).reset_index()

        fig = px.sunburst(industry_job_data,
                         path=['Industry', 'Job_Title'],
                         values='Job_Openings',
                         color='Salary_USD',
                         title='Industry and Job Distribution with Salary Information',
                         color_continuous_scale='Viridis')
        
        fig.update_layout(height=600)
        return fig
    except Exception as e:
        st.error(f"Error in industry distribution chart: {str(e)}")
        return None

@st.cache_data
def create_skill_demand_heatmap(filtered_df):
    """Create heatmap for skill demand analysis"""
    try:
        heatmap_data = pd.pivot_table(
            filtered_df,
            values='Job_Openings',
            index='Industry',
            columns='Experience_Level',
            aggfunc='sum',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            text=heatmap_data.values.astype(int),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title='Job Demand Heatmap: Industry vs Experience Level',
            height=500,
            xaxis_title='Experience Level',
            yaxis_title='Industry'
        )
        return fig
    except Exception as e:
        st.error(f"Error in skill demand heatmap: {str(e)}")
        return None

@st.cache_data
def create_remote_work_analysis(filtered_df):
    """Create stacked bar chart for remote work analysis"""
    try:
        remote_data = filtered_df.groupby(['Industry', 'Remote_Work']).agg({
            'Job_Openings': 'sum'
        }).reset_index()

        fig = px.bar(remote_data,
                    x='Industry',
                    y='Job_Openings',
                    color='Remote_Work',
                    title='Remote Work Distribution by Industry',
                    barmode='stack')
        
        fig.update_layout(
            height=500,
            xaxis_title='Industry',
            yaxis_title='Number of Job Openings',
            xaxis_tickangle=-45
        )
        return fig
    except Exception as e:
        st.error(f"Error in remote work analysis: {str(e)}")
        return None

def main():
    # Title and introduction
    st.title("🎯 AI and Tech Job Market Analysis Dashboard")
    
    # Load data
    with st.spinner('Loading and processing data...'):
        df = load_and_clean_data()
    
    if df is None:
        st.error("Failed to load data. Please check your dataset.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max()))
    )
    
    job_cats = st.sidebar.multiselect(
        "Select Job Categories",
        options=df['Job_Title'].unique(),
        default=df['Job_Title'].unique()
    )
    
    industries = st.sidebar.multiselect(
        "Select Industries",
        options=df['Industry'].unique(),
        default=df['Industry'].unique()
    )

    # Filter data
    mask = (
        (df['Year'].between(year_range[0], year_range[1])) &
        (df['Job_Title'].isin(job_cats)) &
        (df['Industry'].isin(industries))
    )
    filtered_df = df[mask].copy()

    # Display warning if no data matches filters
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_salary = filtered_df['Salary_USD'].mean()
        st.metric("Average Salary", f"${avg_salary:,.0f}")
    with col2:
        avg_growth = filtered_df['Growth_Rate'].mean()
        st.metric("Growth Rate", f"{avg_growth:.1f}%")
    with col3:
        total_jobs = filtered_df['Job_Openings'].sum()
        st.metric("Total Jobs", f"{total_jobs:,}")
    with col4:
        avg_diversity = filtered_df['Gender_Diversity_Index'].mean()
        st.metric("Diversity Index", f"{avg_diversity:.2f}")

    # Create tabs for visualization sections
    tab1, tab2, tab3 = st.tabs([
        "Market Trends", 
        "Salary Analysis", 
        "Industry Insights"
    ])

    with tab1:
        st.header("📈 Market Trends")
        trend_chart = create_salary_trend_chart(filtered_df)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)

    with tab2:
        st.header("💰 Salary Analysis")
        col1, col2 = st.columns(2)
        with col1:
            scatter_chart = create_salary_experience_scatter(filtered_df)
            if scatter_chart:
                st.plotly_chart(scatter_chart, use_container_width=True)
        with col2:
            box_plot = create_salary_boxplot(filtered_df)
            if box_plot:
                st.plotly_chart(box_plot, use_container_width=True)

    with tab3:
        st.header("🏢 Industry Insights")
        sunburst = create_industry_job_distribution(filtered_df)
        if sunburst:
            st.plotly_chart(sunburst, use_container_width=True)
            
        col1, col2 = st.columns(2)
        with col1:
            heatmap = create_skill_demand_heatmap(filtered_df)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
        with col2:
            remote_chart = create_remote_work_analysis(filtered_df)
            if remote_chart:
                st.plotly_chart(remote_chart, use_container_width=True)

    # Key Findings
    st.header("📋 Key Findings")
    st.markdown("""
    ### Key Insights:
    1. **Salary Trends**:
       - Clear correlation between experience level and salary
       - Significant salary variations across job titles
       - Experience level impacts compensation substantially
       
    2. **Industry Insights**:
       - Varying job distribution across industries
       - Different remote work preferences by sector
       - Distinct skill demand patterns
       
    3. **Market Dynamics**:
       - Industry-specific growth patterns
       - Diverse employment opportunities
       - Clear experience-based progression
    """)

    # Raw Data View
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)

if __name__ == "__main__":
    main()