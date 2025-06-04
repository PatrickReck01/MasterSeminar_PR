import plotly.graph_objects as go
import networkx as nx


def visualize_kg(G, plot_image_save = None):
    pos = nx.spring_layout(G)

    # Edge lines
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Edge hover markers (invisible but interactive)
    edge_hover_x = []
    edge_hover_y = []
    edge_hover_text = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        label = data.get("label", "")
        edge_hover_x.append((x0 + x1) / 2)
        edge_hover_y.append((y0 + y1) / 2)
        edge_hover_text.append(f"{u} → {label} → {v}")

    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode='markers',
        hoverinfo='text',
        text=edge_hover_text,
        marker=dict(size=2, color='rgba(0,0,0,0)')  # Invisible
    )

    # Node positions and labels
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        textfont=dict(size=14, color='black', family='Arial Black'),
        marker=dict(
            showscale=False,
            color='lightblue',
            size=20,
            line_width=2
        )
    )

    # Optional edge labels as static text (can be removed)
    edge_labels = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        label_x = (x0 + x1) / 2
        label_y = (y0 + y1) / 2
        label = data.get("label", "")
        edge_labels.append(go.Scatter(
            x=[label_x],
            y=[label_y],
            text=[label],
            mode='text',
            hoverinfo='none',
            textfont=dict(size=12, color='gray')
        ))

    layout = go.Layout(
        title=dict(text='Knowledge Graph', font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    )

    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace] + edge_labels, layout=layout)
    
    # Save the figure as a png if requested
    if plot_image_save:
        # Save plot to png file
        fig.write_image(plot_image_save, scale=2)
    
    # Show the figure
    fig.show()
    
    return fig




if __name__ == "__main__":
    # Create a sample knowledge graph
    G = nx.DiGraph()
    G.add_edge("SNP", "genome", label="occurs in")
    G.add_edge("SNP", "health", label="has effect on")
    G.add_edge("rs1801133", "cardiovascular disease", label="associated with")
    G.add_edge("MTHFR gene", "rs1801133", label="contains SNP")
    G.add_edge("rs1801133", "folate metabolism", label="impacts")

    # Visualize using Plotly
    fig = visualize_kg(G, plot_image_save="knowledge_graph.png")

    