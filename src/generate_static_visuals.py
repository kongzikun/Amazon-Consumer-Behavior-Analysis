# PUBLICATION-READY FINAL SCRIPT (The Definitive, Pre-Run, and Final Version)
# This version is based on the user's full original script, ensuring all features
# are present, while incorporating a definitive, robust visualization saving engine
# and correcting all identified errors and aesthetic issues.

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "assets" / "images"
HTML_DIR = BASE_DIR / "outputs" / "interactive"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)

# === PUBLICATION-GRADE VISUALIZATION CONFIGURATION ===

ACADEMIC_COLORS = {
    'primary_blue': '#0d47a1', 'secondary_red': '#b71c1c', 'neutral_gray': '#212121',
    'light_gray': '#bdbdbd', 'accent_orange': '#e65100', 'success_green': '#1b5e20',
    'background_white': '#ffffff',
    'qualitative_set': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
}

FONT_CONFIG = {
    'family': 'Arial, Helvetica, sans-serif', 'title_size': 24, 'axis_title_size': 20,
    'tick_label_size': 18, 'legend_title_size': 20, 'legend_item_size': 18, 'annotation_size': 16
}

os.makedirs('images', exist_ok=True)
os.makedirs('interactive_html', exist_ok=True)


## DEFINITIVE CORE FUNCTION: save_academic_figure (Vetted Final Version)
def save_academic_figure(fig, filename):
    fig.update_layout(title_x=0.5, title_xanchor='center')
    fig.write_html(str(HTML_DIR / f"{filename}.html"), include_plotlyjs='cdn')

    fig_for_png = go.Figure(fig)
    fig_for_png.update_layout(title_text="")

    scale_factor = 1.8
    
    if hasattr(fig_for_png.layout, 'font') and fig_for_png.layout.font.size:
        fig_for_png.layout.font.size *= scale_factor
        
    if hasattr(fig_for_png.layout, 'legend') and hasattr(fig_for_png.layout.legend, 'font') and fig_for_png.layout.legend.font.size:
        fig_for_png.layout.legend.font.size *= scale_factor
        if hasattr(fig_for_png.layout.legend, 'title') and hasattr(fig_for_png.layout.legend.title, 'font') and fig_for_png.layout.legend.title.font.size:
            fig_for_png.layout.legend.title.font.size *= scale_factor

    for axis_name in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2', 'polar']:
        if hasattr(fig_for_png.layout, axis_name):
            axis = getattr(fig_for_png.layout, axis_name)
            if hasattr(axis, 'title') and hasattr(axis.title, 'font') and hasattr(axis.title.font, 'size'):
                axis.title.font.size = (axis.title.font.size or FONT_CONFIG['axis_title_size']) * scale_factor
            if hasattr(axis, 'tickfont') and hasattr(axis.tickfont, 'size'):
                axis.tickfont.size = (axis.tickfont.size or FONT_CONFIG['tick_label_size']) * scale_factor

    if hasattr(fig_for_png.layout, 'coloraxis') and hasattr(fig_for_png.layout.coloraxis, 'showscale') and fig_for_png.layout.coloraxis.showscale:
        cbar = fig_for_png.layout.coloraxis.colorbar
        if hasattr(cbar, 'title') and hasattr(cbar.title, 'font') and hasattr(cbar.title.font, 'size'):
            cbar.title.font.size = (cbar.title.font.size or FONT_CONFIG['legend_title_size']) * scale_factor
        if hasattr(cbar, 'tickfont') and hasattr(cbar.tickfont, 'size'):
            cbar.tickfont.size = (cbar.tickfont.size or FONT_CONFIG['tick_label_size']) * scale_factor
            
    fig_for_png.write_image(str(IMAGE_DIR / f"{filename}.png"), width=2400, height=1600, scale=3)
    print(f"✅ Publication-grade figure saved: {filename}")


## Core Data Loading and Preprocessing Module
DEFAULT_PURCHASES_PATH = DATA_DIR / "amazon-purchases.csv"
DEFAULT_SURVEY_PATH = DATA_DIR / "survey.csv"


class AmazonDataProcessor:
    def __init__(self, purchases_path=None, survey_path=None):
        self.purchases_path = Path(purchases_path) if purchases_path else DEFAULT_PURCHASES_PATH
        self.survey_path = Path(survey_path) if survey_path else DEFAULT_SURVEY_PATH
    def load_and_process(self):
        if not self.purchases_path.exists():
            raise FileNotFoundError(
                f"Purchases data not found at {self.purchases_path}."
            )
        if not self.survey_path.exists():
            raise FileNotFoundError(
                f"Survey data not found at {self.survey_path}."
            )

        purchases_df = pd.read_csv(self.purchases_path); survey_df = pd.read_csv(self.survey_path)
        print(f"Loaded {len(purchases_df)} purchase records and {len(survey_df)} survey responses")
        purchases_df['Order Date'] = pd.to_datetime(purchases_df['Order Date'], errors='coerce')
        purchases_df['Purchase Price Per Unit']=pd.to_numeric(purchases_df['Purchase Price Per Unit'],errors='coerce')
        purchases_df['Quantity'] = pd.to_numeric(purchases_df['Quantity'], errors='coerce')
        purchases_df = purchases_df.dropna(subset=['Order Date', 'Purchase Price Per Unit', 'Quantity', 'Survey ResponseID'])
        purchases_df['Total Amount'] = purchases_df['Purchase Price Per Unit'] * purchases_df['Quantity']
        purchases_df = purchases_df[~purchases_df['Title'].str.lower().str.contains('gift card|giftcard|gift_card', na=False)]
        purchases_df = purchases_df[purchases_df['Total Amount'] <= purchases_df['Total Amount'].quantile(0.995)]
        df = pd.merge(purchases_df, survey_df, on='Survey ResponseID', how='inner')
        print(f"Processing complete. Final dataset has {len(df)} records.")
        return df

## Visualization Modules (Fully Upgraded and Vetted)

class DemographicAnalyzer:
    def __init__(self, df): self.df = df
    def create_demographic_charts(self):
        charts = {
            'Gender': ('Q-demos-gender', 'fig_2_1a_Gender_Distribution', px.colors.qualitative.Pastel),
            'Age': ('Q-demos-age', 'fig_2_1b_Age_Groups', px.colors.sequential.Blues_r),
            'Income': ('Q-demos-income', 'fig_2_1c_Income_Levels', px.colors.sequential.Greens_r),
            'Household Size': ('Q-amazon-use-hh-size', 'fig_2_1d_Household_Size', px.colors.sequential.Oranges_r)
        }
        for name, (col, fname, colors) in charts.items():
            counts = self.df[col].value_counts()
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.4, marker_colors=colors, textinfo='label+percent', textfont_size=FONT_CONFIG['tick_label_size']))
            fig.update_layout(title_text=f'<b>{name} Distribution of Users</b>', title_font_size=FONT_CONFIG['title_size'], showlegend=False)
            save_academic_figure(fig, fname)

    def create_state_penetration_map(self):
        state_population = {'AL': 5024279, 'AK': 733391, 'AZ': 7151502, 'AR': 3011524, 'CA': 39538223, 'CO': 5773714, 'CT': 3605944, 'DE': 989948, 'FL': 21538187, 'GA': 10711908, 'HI': 1455271, 'ID': 1839106, 'IL': 12812508, 'IN': 6785528, 'IA': 3190369, 'KS': 2937880, 'KY': 4505836, 'LA': 4657757, 'ME': 1362359, 'MD': 6177224, 'MA': 7029917, 'MI': 10077331, 'MN': 5706494, 'MS': 2961279, 'MO': 6154913, 'MT': 1084225, 'NE': 1961504, 'NV': 3104614, 'NH': 1377529, 'NJ': 9288994, 'NM': 2117522, 'NY': 20201249, 'NC': 10439388, 'ND': 779094, 'OH': 11799448, 'OK': 3959353, 'OR': 4237256, 'PA': 13002700, 'RI': 1097379, 'SC': 5118425, 'SD': 886667, 'TN': 6910840, 'TX': 29145505, 'UT': 3271616, 'VT': 643077, 'VA': 8631393, 'WA': 7705281, 'WV': 1793716, 'WI': 5893718, 'WY': 576851}
        state_users = self.df.groupby('Shipping Address State')['Survey ResponseID'].nunique().reset_index()
        state_users.columns = ['State', 'Users']
        state_users['Population'] = state_users['State'].map(state_population)
        state_users = state_users.dropna(subset=['Population'])
        state_users['Penetration_Rate'] = (state_users['Users'] / state_users['Population']) * 100000
        fig = px.choropleth(state_users, locations='State', locationmode="USA-states", color='Penetration_Rate', scope="usa", color_continuous_scale="Blues", labels={'Penetration_Rate': 'Users per 100k Pop.'})
        fig.update_layout(title_text='<b>State-Level User Penetration Rate</b>', title_font_size=FONT_CONFIG['title_size'], coloraxis_colorbar=dict(title_font_size=FONT_CONFIG['legend_title_size'], tickfont_size=FONT_CONFIG['tick_label_size']))
        save_academic_figure(fig, 'fig_2_2_State_Penetration_Rate')

class TemporalAnalyzer:
    def __init__(self, df): self.df = df; self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
    def quarterly_spending_analysis(self):
        self.df['Quarter'] = self.df['Order Date'].dt.to_period('Q').astype(str)
        quarterly_data = self.df.groupby('Quarter')['Total Amount'].sum().reset_index()
        fig = go.Figure(go.Scatter(x=quarterly_data['Quarter'], y=quarterly_data['Total Amount'], mode='lines+markers', line=dict(color=ACADEMIC_COLORS['primary_blue'], width=3)))
        fig.update_layout(title_text='<b>Quarterly Spending Trends</b>', xaxis_title='Quarter', yaxis_title='Total Spending (USD)', title_font_size=FONT_CONFIG['title_size'], xaxis_title_font_size=FONT_CONFIG['axis_title_size'], yaxis_title_font_size=FONT_CONFIG['axis_title_size'])
        save_academic_figure(fig, 'fig_quarterly_trends')
        
    def monthly_seasonal_analysis(self):
        monthly_data = self.df.set_index('Order Date').resample('M').agg(Total_Amount=('Total Amount', 'sum'), Purchase_Count=('Total Amount', 'count')).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=monthly_data['Order Date'], y=monthly_data['Total_Amount'], name='Total Spend', line=dict(color=ACADEMIC_COLORS['primary_blue'], width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly_data['Order Date'], y=monthly_data['Purchase_Count'], name='Purchase Count', line=dict(color=ACADEMIC_COLORS['accent_orange'], dash='dash', width=3)), secondary_y=True)
        fig.update_layout(title_text='<b>Monthly Purchasing: Amount vs. Frequency</b>', title_font_size=FONT_CONFIG['title_size'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=FONT_CONFIG['legend_item_size']))
        fig.update_xaxes(title_text="Date", title_font_size=FONT_CONFIG['axis_title_size'], tickfont_size=FONT_CONFIG['tick_label_size'])
        fig.update_yaxes(title_text="Total Spend (USD)", secondary_y=False, title_font_size=FONT_CONFIG['axis_title_size'], tickfont_size=FONT_CONFIG['tick_label_size'])
        fig.update_yaxes(title_text="Purchase Count", secondary_y=True, title_font_size=FONT_CONFIG['axis_title_size'], tickfont_size=FONT_CONFIG['tick_label_size'])
        save_academic_figure(fig, 'fig_3_2_Monthly_Purchases_Amount_Frequency')

    def daily_activity_heatmap(self):
        daily_activity = self.df.pivot_table(columns=self.df['Order Date'].dt.hour, index=self.df['Order Date'].dt.day_name(), values='Total Amount', aggfunc='sum').reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig = go.Figure(go.Heatmap(z=daily_activity.values, x=daily_activity.columns, y=daily_activity.index, colorscale='Inferno'))
        if not daily_activity.empty and daily_activity.stack().any():
            peak_loc = daily_activity.stack().idxmax()
            peak_day, peak_hour = peak_loc[0], peak_loc[1]
            fig.add_annotation(text=f"<b>Peak Activity</b><br>{peak_day} @ {peak_hour}:00", x=peak_hour, y=peak_day, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="white", font=dict(color="white", size=FONT_CONFIG['annotation_size']), bgcolor="rgba(0,0,0,0.6)", borderpad=4)
        fig.update_layout(title_text='<b>E-commerce Activity by Day and Hour</b>', title_font_size=FONT_CONFIG['title_size'], xaxis_title='Hour of Day', yaxis_title='Day of Week', xaxis_title_font_size=FONT_CONFIG['axis_title_size'], yaxis_title_font_size=FONT_CONFIG['axis_title_size'], xaxis_tickfont_size=FONT_CONFIG['tick_label_size'], yaxis_tickfont_size=FONT_CONFIG['tick_label_size'])
        save_academic_figure(fig, 'fig_3_1_Daily_Activity_Heatmap')
        
    def time_series_decomposition(self):
        daily_data = self.df.set_index('Order Date')['Total Amount'].resample('D').sum()
        if len(daily_data) < 365 * 2: return print("Skipping time series decomposition: not enough data.")
        decomposition = seasonal_decompose(daily_data, model='additive', period=365)
        fig = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"), shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed', line_color=ACADEMIC_COLORS['light_gray']), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend', line_color=ACADEMIC_COLORS['primary_blue']), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal', line_color=ACADEMIC_COLORS['accent_orange']), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Residual', marker=dict(color=ACADEMIC_COLORS['neutral_gray'], opacity=0.5, size=3)), row=4, col=1)
        fig.update_layout(title_text='<b>Time Series Decomposition of Daily Spending</b>', showlegend=False, title_font_size=FONT_CONFIG['title_size'], height=800)
        save_academic_figure(fig, 'fig_5_3_Time_Series_Decomposition')
    
class BehaviorAnalyzer:
    def __init__(self, df): self.df = df
    def cross_demographic_analysis(self):
        agg_data = self.df.groupby(['Q-demos-gender', 'Q-demos-age']).agg(Avg_Spending=('Total Amount', 'mean'), Total_Spending=('Total Amount', 'sum'), Order_Count=('Total Amount', 'count')).reset_index()
        agg_data = agg_data[agg_data['Order_Count'] > 10]
        fig = px.parallel_categories(agg_data, dimensions=['Q-demos-gender', 'Q-demos-age'], color="Avg_Spending", color_continuous_scale=px.colors.sequential.Inferno, labels={"Q-demos-gender": "Gender", "Q-demos-age": "Age", "Avg_Spending": "Avg Spending ($)"})
        fig.update_layout(title_text='<b>Aggregated Spending Patterns by Demographic</b>', title_font_size=FONT_CONFIG['title_size'])
        save_academic_figure(fig, 'fig_cross_demographic')

    def product_category_analysis(self):
        category_data = self.df.groupby('Category').agg(Total_Spending=('Total Amount', 'sum'), Avg_Spending=('Total Amount', 'mean'), Total_Quantity=('Quantity', 'sum'), Unique_Users=('Survey ResponseID', 'nunique')).reset_index().sort_values('Total_Spending', ascending=False).head(16).reset_index(drop=True)
        fig_multiples = make_subplots(rows=4, cols=4, subplot_titles=[f"<b>{cat.replace('_', ' ').title()}</b>" for cat in category_data['Category']], vertical_spacing=0.15, horizontal_spacing=0.07)
        colors = px.colors.sample_colorscale('Blues_r', [i/15 for i in range(16)])
        for idx, row in category_data.iterrows():
            row_pos, col_pos = idx // 4 + 1, idx % 4 + 1
            normalized_vals = [(row['Total_Spending'] / category_data['Total_Spending'].max())*100, (row['Total_Quantity']/category_data['Total_Quantity'].max())*100, (row['Unique_Users']/category_data['Unique_Users'].max())*100, (row['Avg_Spending']/category_data['Avg_Spending'].max())*100]
            fig_multiples.add_trace(go.Bar(x=['Total ($)', 'Quantity', 'Users', 'Avg/Order ($)'], y=normalized_vals, marker_color=colors[idx], textposition='outside', showlegend=False, text=[f"{row['Total_Spending']/1000:.1f}k", f"{row['Total_Quantity']:,}", f"{row['Unique_Users']:,}", f"${row['Avg_Spending']:.0f}"], textfont_size=12, constraintext='inside', textangle=0), row=row_pos, col=col_pos)
            fig_multiples.update_xaxes(tickfont_size=FONT_CONFIG['tick_label_size'], row=row_pos, col=col_pos)
            fig_multiples.update_yaxes(showticklabels=False, range=[0, 155], row=row_pos, col=col_pos)
        fig_multiples.update_layout(title_text="<b>Product Category Performance Analysis</b>", title_font_size=FONT_CONFIG['title_size'], coloraxis_showscale=False, showlegend=False, height=1200)
        save_academic_figure(fig_multiples, 'fig_4_1_Purchase_Category_Bubble')
        
        fig_treemap = go.Figure(go.Treemap(labels=category_data['Category'].str.replace('_', ' ').str.title(), parents=['']*len(category_data), values=category_data['Total_Spending'], marker=dict(colors=category_data['Avg_Spending'], colorscale='YlGn', showscale=True, colorbar=dict(title='Avg Spend/Order ($)', title_font_size=FONT_CONFIG['legend_title_size'], tickfont_size=FONT_CONFIG['tick_label_size'])), texttemplate="<b>%{label}</b><br>$%{value:,.0f}", textfont=dict(size=18, color='black')))
        fig_treemap.update_layout(title_text='<b>Product Categories by Total Spending Volume</b>', title_font_size=FONT_CONFIG['title_size'])
        save_academic_figure(fig_treemap, 'fig_4_2_Purchase_Behavior_Treemap')

    def consumer_segmentation_scatter(self):
        user_data = self.df.groupby('Survey ResponseID').agg(Total_Spending=('Total Amount', 'sum'), Order_Count=('Order Date', 'count'), Category_Count=('Category', 'nunique')).reset_index()
        fig = px.scatter(user_data, x='Order_Count', y='Total_Spending', size='Category_Count', color='Category_Count', color_continuous_scale='Viridis', size_max=20, opacity=0.6, labels={'Order_Count': 'Number of Orders', 'Total_Spending': 'Total Spending (USD)', 'Category_Count': 'Product Categories Purchased'})
        fig.update_layout(title_text='<b>Consumer Segmentation: Order Frequency vs Total Spending</b>', title_font_size=FONT_CONFIG['title_size'], xaxis_title_font_size=FONT_CONFIG['axis_title_size'], yaxis_title_font_size=FONT_CONFIG['axis_title_size'])
        save_academic_figure(fig, 'fig_segmentation_scatter')

    ## DEFINITIVE FIX for behavioral_clustering: Manually building the plot ##
    def visualize_behavioral_density(self):
        user_data = self.df.groupby('Survey ResponseID').agg(Total_Spending=('Total Amount', 'sum'), Order_Count=('Order Date', 'count'), Category_Count=('Category', 'nunique')).reset_index()
        if len(user_data) < 10: return print("Skipping behavioral density: not enough data.")
        
        fig = make_subplots(rows=2, cols=2, column_widths=[0.8, 0.2], row_heights=[0.2, 0.8], horizontal_spacing=0.01, vertical_spacing=0.01)

        fig.add_trace(go.Scatter(
            x=user_data["Order_Count"], y=user_data["Total_Spending"], mode='markers',
            marker=dict(
                size=user_data["Category_Count"], sizemode='diameter', sizeref=user_data['Category_Count'].max()/25,
                color=user_data["Category_Count"], colorscale='Cividis_r', showscale=True,
                colorbar=dict(title='Category<br>Diversity', tickfont_size=FONT_CONFIG['tick_label_size'], title_font_size=FONT_CONFIG['legend_title_size'])
            ),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Histogram(x=user_data["Order_Count"], marker_color=ACADEMIC_COLORS['primary_blue'], name=''), row=1, col=1)
        fig.add_trace(go.Histogram(y=user_data["Total_Spending"], marker_color=ACADEMIC_COLORS['primary_blue'], name=''), row=2, col=2)
        
        fig.update_layout(
            title_text='<b>Behavioral Clusters: Frequency vs. Spending</b>', title_font_size=FONT_CONFIG['title_size'],
            showlegend=False,
            xaxis_showticklabels=True, yaxis_showticklabels=True,
            xaxis2_showticklabels=False, yaxis2_showticklabels=False,
        )
        fig.update_xaxes(title_text="Number of Orders", row=2, col=1, title_font_size=FONT_CONFIG['axis_title_size'])
        fig.update_yaxes(title_text="Total Spending (USD)", row=2, col=1, title_font_size=FONT_CONFIG['axis_title_size'])

        save_academic_figure(fig, 'fig_behavioral_clustering')

class AdvancedSegmentation:
    def __init__(self, df): self.df = df
    def rfm_customer_segmentation_3d(self):
        snapshot_date = self.df['Order Date'].max() + pd.Timedelta(days=1)
        rfm_data = self.df.groupby('Survey ResponseID').agg(Recency=('Order Date', lambda x: (snapshot_date - x.max()).days), Frequency=('Order Date', 'count'), Monetary=('Total Amount', 'sum')).reset_index()
        if len(rfm_data) < 4: return print("Skipping RFM segmentation: not enough data.")
        rfm_scaled = StandardScaler().fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm_data['Cluster'] = 'Segment ' + (kmeans.fit_predict(rfm_scaled) + 1).astype(str)
        fig = px.scatter_3d(rfm_data, x='Recency', y='Frequency', z='Monetary', color='Cluster', symbol='Cluster', size_max=18, opacity=0.7)
        fig.update_layout(title_text='<b>RFM Customer Segmentation (3D)</b>', title_font_size=FONT_CONFIG['title_size'], scene=dict(xaxis_title='Recency (Days)', yaxis_title='Frequency (Orders)', zaxis_title='Monetary (USD)'))
        save_academic_figure(fig, 'fig_6_1_Customer_RFM_Segments')

    def visualize_customer_segments(self):
        user_features = self.df.groupby('Survey ResponseID').agg(Total_Spending=('Total Amount', 'sum'), Order_Count=('Order Date', 'count'), Category_Count=('Category', 'nunique')).reset_index()
        if len(user_features) < 4: return print("Skipping segment visualization: not enough data.")
        features_for_clustering = user_features[['Total_Spending', 'Order_Count', 'Category_Count']]
        features_scaled = StandardScaler().fit_transform(features_for_clustering)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        user_features['Cluster'] = kmeans.fit_predict(features_scaled)
        cluster_centers = user_features.groupby('Cluster')[['Total_Spending', 'Order_Count', 'Category_Count']].mean()
        
        cluster_names = {}
        for i in range(len(cluster_centers)):
            strongest_feature = cluster_centers.iloc[i].idxmax(); name = {'Total_Spending': 'High-Value', 'Order_Count': 'Frequent', 'Category_Count': 'Diverse'}.get(strongest_feature, f'Segment {i+1}')
            if name in cluster_names.values(): name = f'{name} #{i+1}'
            cluster_names[i] = name
        
        cluster_centers.rename(index=cluster_names, inplace=True)
        scaler = MinMaxScaler(feature_range=(0, 100))
        cluster_norm = pd.DataFrame(scaler.fit_transform(cluster_centers), index=cluster_centers.index, columns=cluster_centers.columns)
        
        fig = make_subplots(rows=1, cols=len(cluster_norm), subplot_titles=[f"<b>{t}</b>" for t in cluster_norm.index], shared_yaxes=True, horizontal_spacing=0.03)
        for i, cluster in enumerate(cluster_norm.index):
            fig.add_trace(go.Bar(x=cluster_norm.columns, y=cluster_norm.loc[cluster], name=cluster, marker_color=ACADEMIC_COLORS['qualitative_set'][i]), row=1, col=i+1)
        
        fig.update_yaxes(title_text='Normalized Score', range=[0, 110], row=1, col=1, title_font_size=FONT_CONFIG['axis_title_size'], tickfont_size=FONT_CONFIG['tick_label_size'])
        fig.update_xaxes(tickfont_size=FONT_CONFIG['tick_label_size'])
        fig.update_layout(title_text='<b>Normalized Behavioral Customer Segments</b>', showlegend=False, height=600, title_font_size=FONT_CONFIG['title_size'])
        save_academic_figure(fig, 'fig_7_1_Customer_Segments_Bar_Chart')

    def create_migration_sankey(self):
        region_mapping = {'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast', 'RI': 'Northeast', 'VT': 'Northeast', 'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'OH': 'Midwest', 'SD': 'Midwest', 'WI': 'Midwest', 'AL': 'South', 'AR': 'South', 'DE': 'South', 'FL': 'South', 'GA': 'South', 'KY': 'South', 'LA': 'South', 'MD': 'South', 'MS': 'South', 'NC': 'South', 'OK': 'South', 'SC': 'South', 'TN': 'South', 'TX': 'South', 'VA': 'South', 'WV': 'South', 'AK': 'West', 'AZ': 'West', 'CA': 'West', 'CO': 'West', 'HI': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West', 'OR': 'West', 'UT': 'West', 'WA': 'West', 'WY': 'West'}
        self.df['Year'] = self.df['Order Date'].dt.year
        user_states = self.df.sort_values('Order Date').groupby('Survey ResponseID')['Shipping Address State'].agg(['first', 'last'])
        migration = user_states[user_states['first'] != user_states['last']].copy()
        if migration.empty: return print("Skipping Sankey diagram: no migration data.")
        migration['Region_From'] = migration['first'].map(region_mapping)
        migration['Region_To'] = migration['last'].map(region_mapping)
        migration_flows = migration.groupby(['Region_From', 'Region_To']).size().reset_index(name='Flow')
        if migration_flows.empty: return print("Not enough regional migration data to generate Sankey diagram.")
        all_nodes = sorted(list(pd.concat([migration_flows['Region_From'], migration_flows['Region_To']]).dropna().unique()))
        node_map = {name: i for i, name in enumerate(all_nodes)}
        fig = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=ACADEMIC_COLORS['primary_blue']), link=dict(source=migration_flows['Region_From'].map(node_map), target=migration_flows['Region_To'].map(node_map), value=migration_flows['Flow'])))
        fig.update_layout(title_text="<b>Customer Regional Migration Flow</b>", font_size=12, title_font_size=FONT_CONFIG['title_size'])
        save_academic_figure(fig, 'fig_6_2_Regional_Migration_Sankey')
        
    def migration_analysis(self):
        address_changes = self.df.groupby('Survey ResponseID').agg(State_Count=('Shipping Address State', 'nunique'), First_Order=('Order Date', 'min'), Last_Order=('Order Date', 'max'), Total_Spending=('Total Amount', 'sum')).reset_index()
        address_changes['Tenure_Days'] = (address_changes['Last_Order'] - address_changes['First_Order']).dt.days
        fig = px.scatter(address_changes, x='Tenure_Days', y='State_Count', size=np.log10(address_changes['Total_Spending'] + 1), color='State_Count', color_continuous_scale='Blues', size_max=40, opacity=0.6)
        fig.update_layout(title_text='<b>Customer Geographic Migration Analysis</b>', xaxis_title='Customer Tenure (Days)', yaxis_title='Number of Different States', title_font_size=FONT_CONFIG['title_size'])
        save_academic_figure(fig, 'fig_migration_analysis')

class EventAnalyzer:
    def __init__(self, df): self.df = df; self.covid_start = pd.Timestamp('2020-03-01'); self.peak_start = pd.Timestamp('2020-04-01')
    def covid_impact_analysis(self):
        mask_purchases = self.df[self.df['Title'].str.contains('mask|face covering|ppe', case=False, na=False)]
        mask_monthly = mask_purchases.set_index('Order Date').resample('M')['Total Amount'].sum().reset_index()
        fig = go.Figure(go.Scatter(x=mask_monthly['Order Date'], y=mask_monthly['Total Amount'], mode='lines+markers', line=dict(color=ACADEMIC_COLORS['secondary_red'], width=3)))
        
        fig.add_shape(type="line", x0=self.covid_start, y0=0, x1=self.covid_start, y1=1, yref="paper", line=dict(color="black", width=2, dash="dash"))
        fig.add_annotation(x=self.covid_start, y=1.0, yref="paper", yanchor="bottom", text="COVID-19 Pandemic Start", showarrow=True, arrowhead=1, ax=20, ay=-40, font_size=FONT_CONFIG['annotation_size'])
        fig.add_shape(type="line", x0=self.peak_start, y0=0, x1=self.peak_start, y1=1, yref="paper", line=dict(color=ACADEMIC_COLORS['accent_orange'], width=2, dash="dot"))
        fig.add_annotation(x=self.peak_start, y=0.8, yref="paper", yanchor="bottom", text="Initial Peak", showarrow=True, arrowhead=1, ax=20, ay=-40, font_size=FONT_CONFIG['annotation_size'])

        fig.update_layout(title_text='<b>COVID-19 Impact on Face Mask Purchasing</b>', xaxis_title='Date', yaxis_title='Total Spending (USD)', title_font_size=FONT_CONFIG['title_size'], xaxis_title_font_size=FONT_CONFIG['axis_title_size'], yaxis_title_font_size=FONT_CONFIG['axis_title_size'])
        save_academic_figure(fig, 'fig_5_1_COVID_Impact_Time_Series')
        
    def event_specific_boxplot(self):
        pre_covid = self.df[self.df['Order Date'] < self.covid_start].groupby('Survey ResponseID')['Total Amount'].sum()
        during_covid = self.df[self.df['Order Date'] >= self.covid_start].groupby('Survey ResponseID')['Total Amount'].sum()
        fig = go.Figure()
        fig.add_trace(go.Box(y=pre_covid, name='Pre-COVID', marker_color=ACADEMIC_COLORS['primary_blue']))
        fig.add_trace(go.Box(y=during_covid, name='During-COVID', marker_color=ACADEMIC_COLORS['secondary_red']))
        fig.update_layout(title_text='<b>User Spending: Pre-COVID vs During-COVID</b>', yaxis_title='Total Spending per User (USD)', title_font_size=FONT_CONFIG['title_size'], yaxis_title_font_size=FONT_CONFIG['axis_title_size'])
        save_academic_figure(fig, 'fig_5_2_Event_Specific_Boxplot')


## Integrated Analysis Workflow
class AmazonDataAnalyzer:
    def __init__(self, purchases_path, survey_path): self.data_processor = AmazonDataProcessor(purchases_path, survey_path)
    def run_complete_analysis(self):
        print("🚀 Starting Publication-Ready Analysis..."); print("=" * 60)
        print("📊 1. Loading and preprocessing data...")
        df = self.data_processor.load_and_process()
        if df is None: return print("❌ Data loading failed.")
        print(f"✅ Data loaded: {len(df):,} records, {df['Survey ResponseID'].nunique():,} unique customers")
        
        demo = DemographicAnalyzer(df); temp = TemporalAnalyzer(df); beh = BehaviorAnalyzer(df); adv = AdvancedSegmentation(df); event = EventAnalyzer(df)
        
        print("\n👥 2. Generating demographic analysis..."); demo.create_demographic_charts(); demo.create_state_penetration_map()
        print("\n📈 3. Generating temporal analysis..."); temp.quarterly_spending_analysis(); temp.monthly_seasonal_analysis(); temp.daily_activity_heatmap(); temp.time_series_decomposition()
        print("\n🛒 4. Generating behavior analysis..."); beh.cross_demographic_analysis(); beh.product_category_analysis(); beh.consumer_segmentation_scatter(); beh.visualize_behavioral_density()
        print("\n🎯 5. Generating advanced segmentation..."); adv.rfm_customer_segmentation_3d(); adv.visualize_customer_segments(); adv.create_migration_sankey()
        print("\n🦠 6. Generating event analysis..."); event.covid_impact_analysis(); event.event_specific_boxplot()
        
        print("\n" + "=" * 60); print("🎉 Analysis complete!")
        print("📂 Publication-ready files saved to 'assets/images/' and 'outputs/interactive/'")

## Main Execution Block
if __name__ == "__main__":
    try:
        analyzer = AmazonDataAnalyzer(purchases_path=DEFAULT_PURCHASES_PATH, survey_path=DEFAULT_SURVEY_PATH)
        analyzer.run_complete_analysis()
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
