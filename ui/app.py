import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Configure page
st.set_page_config(
    page_title="Big Data Explorer",
    page_icon="🔍",
    layout="wide"
)

# Create sample data
def create_sample_data():
    data = {
        'age': [39, 50, 38, 53, 28, 37, 49, 52, 31, 42],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private',
                     'Private', 'Private', 'Self-emp-not-inc', 'Private', 'Private'],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors',
                     'Masters', '9th', 'HS-grad', 'Masters', 'Bachelors'],
        'income': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '>50K', '>50K', '>50K']
    }
    return pd.DataFrame(data)

def create_fifa_data():
    data = {
        'Player': ['Miroslav Klose', 'Ronaldo', 'Gerd Müller', 'Just Fontaine', 'Pelé'],
        'Goals': [16, 15, 14, 13, 12],
        'Matches': [24, 19, 13, 6, 14],
        'Country': ['Germany', 'Brazil', 'Germany', 'France', 'Brazil']
    }
    return pd.DataFrame(data)

# SIDEBAR: Team Profiles
st.sidebar.markdown("## 🧠 Project Contributors")
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 15px;'>
    <strong>Victoria Love Franklin</strong><br>
    <small>Scientist, U.S. Dept. of Defense</small>
</div>
<div style='text-align: center; margin-bottom: 15px;'>
    <strong>Michael J. Paul</strong><br>
    <small>Professor, Data Science & Informatics</small>
</div>
<div style='text-align: center; margin-bottom: 15px;'>
    <strong>Emirrah Sanders</strong><br>
    <small>Data Scientist, Public Health Analytics</small>
</div>
<div style='text-align: center; margin-bottom: 15px;'>
    <strong>Courtney Quarterman</strong><br>
    <small>AI & Data Science Educator</small>
</div>
""", unsafe_allow_html=True)

# MAIN APP
st.title("Big Data Explorer")

# Technology selector
tech_choice = st.sidebar.radio("Choose Technology", ["Apache Spark", "Graph Database (Neo4j)"])

# APACHE SPARK SECTION
if tech_choice == "Apache Spark":
    st.header("Apache Spark Operations")

    # Data source selection
    data_source = st.radio("Select Data Source", ["Sample Adult Dataset", "Upload CSV"])

    if data_source == "Sample Adult Dataset":
        df = create_sample_data()
        st.success("✅ Sample Adult Dataset loaded!")
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
        else:
            df = None
            st.info("Please upload a CSV file to continue.")

    if df is not None:
        task = st.selectbox("Choose Task", [
            "View Top 5 Rows",
            "Count Age < 40",
            "Count by Education",
            "Basic Statistics",
            "Column Analysis"
        ])

        if task == "View Top 5 Rows":
            st.subheader("📊 Dataset Preview")
            st.dataframe(df.head())

            st.subheader("📋 Schema")
            # Create Spark-like schema output
            schema_lines = ["root"]
            for col in df.columns:
                dtype = str(df[col].dtype)
                # Convert pandas dtypes to Spark-like dtypes
                if dtype == 'int64':
                    spark_type = 'integer'
                elif dtype == 'float64':
                    spark_type = 'double'
                elif dtype == 'object':
                    spark_type = 'string'
                elif dtype == 'bool':
                    spark_type = 'boolean'
                else:
                    spark_type = dtype

                schema_lines.append(f" |-- {col}: {spark_type} (nullable = true)")

            schema_output = "\n".join(schema_lines)
            st.code(schema_output)

        elif task == "Count Age < 40":
            if 'age' in df.columns:
                count = len(df[df['age'] < 40])
                st.success(f"🎯 Number of people below 40 years old: **{count}**")

                fig, ax = plt.subplots()
                df['age'].hist(bins=10, ax=ax)
                ax.axvline(x=40, color='red', linestyle='--', label='Age 40')
                ax.set_title('Age Distribution')
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("❌ No 'age' column found in the dataset")
                st.info("Available columns: " + ", ".join(df.columns))

        elif task == "Count by Education":
            if 'education' in df.columns:
                edu_counts = df['education'].value_counts()
                st.subheader("📚 Education Level Counts")
                st.dataframe(edu_counts)

                fig, ax = plt.subplots()
                edu_counts.plot(kind='bar', ax=ax)
                ax.set_title('Count by Education Level')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.error("❌ No 'education' column found in the dataset")
                st.info("Available columns: " + ", ".join(df.columns))

                # Offer alternative: analyze any categorical column
                cat_columns = df.select_dtypes(include=['object']).columns.tolist()
                if cat_columns:
                    st.info("📊 Try analyzing these categorical columns instead:")
                    selected_col = st.selectbox("Select a column to analyze:", cat_columns)
                    if st.button(f"Analyze {selected_col}"):
                        col_counts = df[selected_col].value_counts().head(10)
                        st.dataframe(col_counts)

                        fig, ax = plt.subplots()
                        col_counts.plot(kind='bar', ax=ax)
                        ax.set_title(f'Top 10 Values in {selected_col}')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

        elif task == "Basic Statistics":
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe(include='all'))

            # Show data info
            st.subheader("📋 Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

        elif task == "Column Analysis":
            st.subheader("🔍 Column Analysis")
            selected_col = st.selectbox("Select column to analyze:", df.columns)

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Analysis of '{selected_col}':**")
                st.write(f"- Data type: {df[selected_col].dtype}")
                st.write(f"- Non-null values: {df[selected_col].count()}")
                st.write(f"- Null values: {df[selected_col].isnull().sum()}")
                st.write(f"- Unique values: {df[selected_col].nunique()}")

            with col2:
                st.write("**Top 10 Values:**")
                value_counts = df[selected_col].value_counts().head(10)
                st.dataframe(value_counts)

            # Visualization based on data type
            if df[selected_col].dtype in ['int64', 'float64']:
                st.subheader("📈 Distribution")
                fig, ax = plt.subplots()
                df[selected_col].hist(bins=20, ax=ax)
                ax.set_title(f'Distribution of {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            else:
                st.subheader("📊 Value Counts")
                fig, ax = plt.subplots()
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Top Values in {selected_col}')
                plt.xticks(rotation=45)
                st.pyplot(fig)

# NEO4J SECTION
elif tech_choice == "Graph Database (Neo4j)":
    st.header("Neo4j Graph Explorer")

    query_option = st.selectbox("Select Query", [
        "Count All Nodes",
        "Create Ronaldinho & Brazil",
        "Top Scorers Chart",
        "Load CSV & Create Graph",
        "FIFA Graph Visualization"
    ])

    if query_option == "Count All Nodes":
        st.code("MATCH (n) RETURN count(n) AS nodeCount", language='cypher')
        st.json([{"nodeCount": 150}])

    elif query_option == "Create Ronaldinho & Brazil":
        query = '''CREATE (p:Player {name: "Ronaldo Gaúcho", YOB: 1980, POB: "Porto Alegre"})
CREATE (c:Country {name: "Brazil"})
CREATE (p)-[:PLAYER_OF]->(c)'''
        st.code(query, language='cypher')
        if st.button("Execute Query"):
            st.success("✅ Nodes and relationship created!")

            # Create an interactive graph visualization
            G = nx.Graph()
            G.add_node("Ronaldo Gaúcho", node_type="Player", color="#FF6B6B")
            G.add_node("Brazil", node_type="Country", color="#4ECDC4")
            G.add_edge("Ronaldo Gaúcho", "Brazil", relationship="PLAYER_OF")

            pos = nx.spring_layout(G, k=2)

            # Create interactive Plotly version
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=4, color='rgba(125,125,125,0.7)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )

            node_x, node_y, node_text, node_color, node_size, hover_text = [], [], [], [], [], []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

                if G.nodes[node]['node_type'] == 'Player':
                    node_color.append('#FF6B6B')
                    node_size.append(50)
                    hover_text.append(f"<b>{node}</b><br>Type: Player<br>Born: 1980<br>Birthplace: Porto Alegre")
                else:
                    node_color.append('#4ECDC4')
                    node_size.append(60)
                    hover_text.append(f"<b>{node}</b><br>Type: Country<br>Region: South America")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                textfont=dict(size=12, color="white", family="Arial Black"),
                hovertext=hover_text,
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=3, color='white'),
                    opacity=0.9
                ),
                showlegend=False
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title=dict(
                    text="Player-Country Relationship Graph",
                    x=0.5,
                    font=dict(size=18)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="🖱️ Try dragging the nodes around!",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.1,
                        xanchor='center', yanchor='bottom',
                        font=dict(color='#666', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                dragmode='pan',
                height=400
            )

            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize'
            }

            st.plotly_chart(fig, use_container_width=True, config=config)

    elif query_option == "Top Scorers Chart":
        st.subheader("⚽ FIFA World Cup Top Scorers")

        fifa_df = create_fifa_data()
        st.dataframe(fifa_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = range(len(fifa_df))

        ax.bar([i - bar_width/2 for i in index], fifa_df['Goals'],
               bar_width, label='Goals', color='skyblue')
        ax.bar([i + bar_width/2 for i in index], fifa_df['Matches'],
               bar_width, label='Matches', color='lightgreen')

        ax.set_xlabel('Players')
        ax.set_ylabel('Count')
        ax.set_title('Goals and Matches by Player')
        ax.set_xticks(index)
        ax.set_xticklabels(fifa_df['Player'], rotation=45)
        ax.legend()

        st.pyplot(fig)

        csv = fifa_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="fifa_top_scorers.csv",
            mime='text/csv'
        )

    elif query_option == "Load CSV & Create Graph":
        st.subheader("📤 Import CSV Data and Create Graph Visualization")

        uploaded_file = st.file_uploader("Upload CSV to create graph", type=["csv"])

        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            st.write("**Preview of data to import:**")
            st.dataframe(df_upload.head())

            # Graph creation options
            st.subheader("🔗 Graph Configuration")

            col1, col2 = st.columns(2)
            with col1:
                node_column = st.selectbox("Select Node Column:", df_upload.columns)
                relationship_type = st.text_input("Relationship Type", value="CONNECTED_TO")

            with col2:
                if len(df_upload.columns) > 1:
                    connect_column = st.selectbox("Connect to Column:",
                                                [col for col in df_upload.columns if col != node_column])
                else:
                    connect_column = None
                    st.info("Upload a CSV with multiple columns to create relationships")

            layout_option = st.selectbox("Graph Layout:",
                                       ["spring", "circular", "kamada_kawai", "random"])

            if st.button("Create Graph Visualization") and connect_column:
                # Create NetworkX graph
                G = nx.Graph()

                # Add nodes and edges
                for _, row in df_upload.iterrows():
                    node1 = str(row[node_column])
                    node2 = str(row[connect_column])

                    G.add_node(node1, node_type=node_column, color="lightblue")
                    G.add_node(node2, node_type=connect_column, color="lightcoral")
                    G.add_edge(node1, node2, relationship=relationship_type)

                # Choose layout
                if layout_option == "spring":
                    pos = nx.spring_layout(G, k=1, iterations=50)
                elif layout_option == "circular":
                    pos = nx.circular_layout(G)
                elif layout_option == "kamada_kawai":
                    pos = nx.kamada_kawai_layout(G)
                else:
                    pos = nx.random_layout(G)

                # Create interactive Plotly visualization
                def create_interactive_graph(G, pos, title):
                    # Prepare edge traces
                    edge_x = []
                    edge_y = []
                    edge_info = []

                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_info.append(edge[0] + " → " + edge[1])

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=2, color='rgba(125,125,125,0.5)'),
                        hoverinfo='none',
                        mode='lines',
                        showlegend=False
                    )

                    # Prepare node traces
                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    node_size = []
                    hover_text = []

                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(str(node))

                        node_type = G.nodes[node].get('node_type')
                        if node_type == node_column:
                            node_color.append('#FF6B6B')  # Red for first column
                            node_size.append(25)
                            hover_text.append(f"<b>{node}</b><br>Type: {node_column}<br>Connections: {G.degree(node)}")
                        else:
                            node_color.append('#4ECDC4')  # Teal for second column
                            node_size.append(25)
                            hover_text.append(f"<b>{node}</b><br>Type: {connect_column}<br>Connections: {G.degree(node)}")

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        hoverinfo='text',
                        text=node_text,
                        textposition="middle center",
                        textfont=dict(size=10, color="white"),
                        hovertext=hover_text,
                        marker=dict(
                            size=node_size,
                            color=node_color,
                            line=dict(width=2, color='white'),
                            opacity=0.9
                        ),
                        showlegend=False
                    )

                    # Create the figure
                    fig = go.Figure(data=[edge_trace, node_trace])

                    fig.update_layout(
                        title=dict(
                            text=title,
                            x=0.5,
                            font=dict(size=20)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[
                            dict(
                                text="💡 Drag nodes to move them around! Use mouse wheel to zoom.",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(color='#666', size=12)
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        # Enable dragging and zooming
                        dragmode='pan',
                        uirevision=True
                    )

                    # Make the graph interactive with custom config
                    config = {
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'scrollZoom': True,
                        'doubleClick': 'reset+autosize'
                    }

                    return fig, config

                graph_title = f"Interactive Graph: {node_column} ↔ {connect_column}"
                fig, config = create_interactive_graph(G, pos, graph_title)
                st.plotly_chart(fig, use_container_width=True, config=config)

                # Show graph statistics
                st.subheader("📊 Graph Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nodes", G.number_of_nodes())
                with col2:
                    st.metric("Edges", G.number_of_edges())
                with col3:
                    st.metric("Density", f"{nx.density(G):.3f}")
                with col4:
                    if nx.is_connected(G):
                        st.metric("Diameter", nx.diameter(G))
                    else:
                        st.metric("Components", nx.number_connected_components(G))

                # Show Cypher query equivalent
                st.subheader("📝 Equivalent Cypher Query:")
                node_col_title = node_column.title()
                connect_col_title = connect_column.title()
                cypher_query = f"""
// Create nodes for {node_column}
LOAD CSV WITH HEADERS FROM 'file:///your_file.csv' AS row
CREATE (a:{node_col_title} {{name: row.{node_column}}})

// Create nodes for {connect_column}
LOAD CSV WITH HEADERS FROM 'file:///your_file.csv' AS row
CREATE (b:{connect_col_title} {{name: row.{connect_column}}})

// Create relationships
LOAD CSV WITH HEADERS FROM 'file:///your_file.csv' AS row
MATCH (a:{node_col_title} {{name: row.{node_column}}})
MATCH (b:{connect_col_title} {{name: row.{connect_column}}})
CREATE (a)-[:{relationship_type}]->(b)"""
                st.code(cypher_query, language='cypher')

        else:
            st.info("Upload a CSV file to create graph visualizations")

    elif query_option == "FIFA Graph Visualization":
        st.subheader("⚽ FIFA Players Interactive Graph Network")

        fifa_df = create_fifa_data()
        st.dataframe(fifa_df)

        # Graph configuration options
        col1, col2 = st.columns(2)
        with col1:
            physics_enabled = st.checkbox("Enable Physics Simulation", value=True)
            show_country_connections = st.checkbox("Connect players from same country", value=True)
        with col2:
            node_size_factor = st.slider("Node Size Factor", min_value=1, max_value=5, value=2)
            edge_opacity = st.slider("Edge Opacity", min_value=0.1, max_value=1.0, value=0.6)

        # Create NetworkX graph
        G = nx.Graph()

        # Add player nodes with attributes
        for _, row in fifa_df.iterrows():
            G.add_node(row['Player'],
                      node_type='Player',
                      goals=row['Goals'],
                      matches=row['Matches'],
                      country=row['Country'],
                      color='#FF6B6B')  # Red for players

        # Add country nodes
        countries = fifa_df['Country'].unique()
        for country in countries:
            G.add_node(country,
                      node_type='Country',
                      color='#4ECDC4')  # Teal for countries

        # Add edges between players and countries
        for _, row in fifa_df.iterrows():
            G.add_edge(row['Player'], row['Country'],
                      relationship='PLAYS_FOR',
                      edge_type='player_country')

        # Add edges between players from same country if enabled
        if show_country_connections:
            for country in countries:
                players_from_country = fifa_df[fifa_df['Country'] == country]['Player'].tolist()
                for i, player1 in enumerate(players_from_country):
                    for player2 in players_from_country[i+1:]:
                        G.add_edge(player1, player2,
                                  relationship='SAME_COUNTRY',
                                  edge_type='player_player')

        # Create sophisticated layout
        if physics_enabled:
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Create the interactive Plotly graph
        def create_fifa_interactive_graph(G, pos):
            # Prepare edge traces with different styles
            edge_traces = []

            # Player-Country edges
            pc_edge_x, pc_edge_y = [], []
            for edge in G.edges():
                edge_data = G.edges[edge]
                if edge_data.get('edge_type') == 'player_country':
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    pc_edge_x.extend([x0, x1, None])
                    pc_edge_y.extend([y0, y1, None])

            if pc_edge_x:
                rgba_color = f'rgba(255,107,107,{edge_opacity})'
                edge_traces.append(go.Scatter(
                    x=pc_edge_x, y=pc_edge_y,
                    line=dict(width=3, color=rgba_color),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False,
                    name='Plays For'
                ))

            # Player-Player edges (same country)
            pp_edge_x, pp_edge_y = [], []
            for edge in G.edges():
                edge_data = G.edges[edge]
                if edge_data.get('edge_type') == 'player_player':
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    pp_edge_x.extend([x0, x1, None])
                    pp_edge_y.extend([y0, y1, None])

            if pp_edge_x:
                rgba_color = f'rgba(78,205,196,{edge_opacity*0.5})'
                edge_traces.append(go.Scatter(
                    x=pp_edge_x, y=pp_edge_y,
                    line=dict(width=1, color=rgba_color, dash='dash'),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False,
                    name='Same Country'
                ))

            # Prepare node traces
            player_x, player_y, player_text, player_hover = [], [], [], []
            country_x, country_y, country_text, country_hover = [], [], [], []
            player_sizes, country_sizes = [], []

            for node in G.nodes():
                x, y = pos[node]
                node_data = G.nodes[node]

                if node_data.get('node_type') == 'Player':
                    player_x.append(x)
                    player_y.append(y)
                    player_text.append(node.split()[0])  # First name only for display

                    goals = node_data.get('goals', 0)
                    matches = node_data.get('matches', 0)
                    country = node_data.get('country', '')
                    player_hover.append(
                        f"<b>{node}</b><br>"
                        f"⚽ Goals: {goals}<br>"
                        f"🏟️ Matches: {matches}<br>"
                        f"🏴 Country: {country}<br>"
                        f"🔗 Connections: {G.degree(node)}"
                    )
                    player_sizes.append(20 + goals * node_size_factor)
                else:
                    country_x.append(x)
                    country_y.append(y)
                    country_text.append(node)

                    # Count players from this country
                    country_players = fifa_df[fifa_df['Country']==node]['Player']
                    num_players = len(country_players)

                    country_hover.append(
                        f"<b>{node}</b><br>"
                        f"🏴 Country<br>"
                        f"👥 Players: {num_players}<br>"
                        f"🔗 Connections: {G.degree(node)}"
                    )
                    country_sizes.append(35)

            # Player nodes
            player_trace = go.Scatter(
                x=player_x, y=player_y,
                mode='markers+text',
                marker=dict(
                    size=player_sizes,
                    color='#FF6B6B',
                    line=dict(width=3, color='white'),
                    opacity=0.9,
                    sizemode='diameter'
                ),
                text=player_text,
                textposition="middle center",
                textfont=dict(size=10, color="white", family="Arial Black"),
                hoverinfo='text',
                hovertext=player_hover,
                name='Players',
                showlegend=True
            )

            # Country nodes
            country_trace = go.Scatter(
                x=country_x, y=country_y,
                mode='markers+text',
                marker=dict(
                    size=country_sizes,
                    color='#4ECDC4',
                    line=dict(width=3, color='white'),
                    opacity=0.9,
                    symbol='diamond'
                ),
                text=country_text,
                textposition="middle center",
                textfont=dict(size=12, color="white", family="Arial Black"),
                hoverinfo='text',
                hovertext=country_hover,
                name='Countries',
                showlegend=True
            )

            # Combine all traces
            all_traces = edge_traces + [player_trace, country_trace]

            # Create the figure
            fig = go.Figure(data=all_traces)

            fig.update_layout(
                title=dict(
                    text="⚽ FIFA Top Scorers Network - Drag Nodes to Explore!",
                    x=0.5,
                    font=dict(size=22, color='#2c3e50')
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=60),
                annotations=[
                    dict(
                        text="🖱️ Click and drag nodes • 🔍 Scroll to zoom • 📱 Double-click to reset",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.05,
                        xanchor='center', yanchor='bottom',
                        font=dict(color='#7f8c8d', size=14)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                dragmode='pan',
                uirevision=True,
                width=None,
                height=600
            )

            return fig

        # Create and display the interactive graph
        fig = create_fifa_interactive_graph(G, pos)

        # Custom configuration for maximum interactivity
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': [
                'drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape'
            ],
            'scrollZoom': True,
            'doubleClick': 'reset+autosize',
            'showTips': True,
            'responsive': True
        }

        st.plotly_chart(fig, use_container_width=True, config=config)

        # Interactive controls
        st.subheader("🎮 Graph Controls")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Regenerate Layout"):
                st.rerun()
        with col2:
            if st.button("📊 Show Adjacency Matrix"):
                adj_matrix = nx.adjacency_matrix(G).todense()
                df_adj = pd.DataFrame(adj_matrix,
                                    index=list(G.nodes()),
                                    columns=list(G.nodes()))
                st.dataframe(df_adj)
        with col3:
            if st.button("📈 Export Graph Data"):
                graph_data = {
                    'nodes': [{'id': node, **G.nodes[node]} for node in G.nodes()],
                    'edges': [{'source': edge[0], 'target': edge[1], **G.edges[edge]} for edge in G.edges()]
                }
                st.json(graph_data)

        # Network analysis
        st.subheader("🔍 Network Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", G.number_of_nodes())
        with col2:
            st.metric("Total Edges", G.number_of_edges())
        with col3:
            st.metric("Network Density", f"{nx.density(G):.3f}")
        with col4:
            st.metric("Avg Clustering", f"{nx.average_clustering(G):.3f}")

        # Centrality measures
        st.subheader("📈 Node Importance Rankings")

        # Calculate different centrality measures
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)

        centrality_data = []
        for node in G.nodes():
            node_data = G.nodes[node]
            centrality_data.append({
                'Node': node,
                'Type': node_data.get('node_type', 'Unknown'),
                'Degree Centrality': degree_cent[node],
                'Betweenness Centrality': betweenness_cent[node],
                'Closeness Centrality': closeness_cent[node]
            })

        centrality_df = pd.DataFrame(centrality_data).sort_values('Degree Centrality', ascending=False)
        st.dataframe(centrality_df, use_container_width=True)

        # Create centrality comparison chart
        fig_cent = px.scatter(
            centrality_df,
            x='Degree Centrality',
            y='Betweenness Centrality',
            color='Type',
            size='Closeness Centrality',
            hover_name='Node',
            title='Node Centrality Comparison',
            color_discrete_map={'Player': '#FF6B6B', 'Country': '#4ECDC4'}
        )
        fig_cent.update_layout(height=400)
        st.plotly_chart(fig_cent, use_container_width=True)

st.markdown("---")
st.markdown("### 🚀 Built with Streamlit for Big Data Exploration")
