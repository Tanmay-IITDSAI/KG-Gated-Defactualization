import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================================================
# JSON FILES
# =========================================================

files = [
    "kg_001_battery_issue_ORD-7741_Priya_Sharma.json",
    "kg_002_battery_issue_ORD-7742_Leo_Chen.json",
    "kg_003_battery_issue_ORD-7743_Aisha_Patel.json"
]

# =========================================================
# NODE COLORS
# =========================================================

NODE_COLORS = {
    "CUSTOMER_NAME": "#f6c85f",
    "ORDER_ID": "#6fa8dc",
    "PRODUCT": "#82e0aa",
    "ISSUE": "#f1948a",
    "URGENCY": "#c39bd3",
    "SENTIMENT": "#85c1e9"
}

# =========================================================
# GENERATE EACH GRAPH SEPARATELY
# =========================================================

for idx, file in enumerate(files):

    # -----------------------------------------------------
    # LOAD JSON
    # -----------------------------------------------------

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    kg = data["knowledge_graph"]

    # -----------------------------------------------------
    # CREATE GRAPH
    # -----------------------------------------------------

    G = nx.DiGraph()

    # Add Nodes
    for node in kg["nodes"]:
        G.add_node(
            node["value"],
            node_type=node["type"],
            salience=node["salience"]
        )

    # Add Edges
    for edge in kg["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            relation=edge["relation"],
            weight=edge["weight"]
        )

    # -----------------------------------------------------
    # FIGURE
    # -----------------------------------------------------

    plt.figure(figsize=(12, 9))

    # Better layout for paper-quality visualization
    pos = nx.spring_layout(
        G,
        seed=42,
        k=2.2
    )

    # -----------------------------------------------------
    # DRAW NODES
    # -----------------------------------------------------

    for node, attrs in G.nodes(data=True):

        node_type = attrs["node_type"]

        color = NODE_COLORS.get(node_type, "#d3d3d3")

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=color,
            node_size=5000,
            edgecolors='black',
            linewidths=2,
            alpha=0.95
        )

    # -----------------------------------------------------
    # DRAW EDGES
    # -----------------------------------------------------

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle='-|>',
        arrowsize=25,
        width=2.5,
        edge_color='black'
    )

    # -----------------------------------------------------
    # WRAP LONG LABELS
    # -----------------------------------------------------

    wrapped_labels = {}

    for node in G.nodes():

        if len(node) > 35:

            wrapped = "\n".join(
                [node[i:i+30] for i in range(0, len(node), 30)]
            )

            wrapped_labels[node] = wrapped

        else:
            wrapped_labels[node] = node

    # -----------------------------------------------------
    # NODE LABELS
    # -----------------------------------------------------

    nx.draw_networkx_labels(
        G,
        pos,
        labels=wrapped_labels,
        font_size=9,
        font_weight='bold'
    )

    # -----------------------------------------------------
    # EDGE LABELS
    # -----------------------------------------------------

    edge_labels = {}

    for u, v, d in G.edges(data=True):

        edge_labels[(u, v)] = (
            f"{d['relation']}\n({d['weight']:.3f})"
        )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    plt.title(
        f"Knowledge Graph for Case {idx+1} ({data['order_id']})",
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    # -----------------------------------------------------
    # LEGEND
    # -----------------------------------------------------

    legend_elements = [
        Patch(
            facecolor=color,
            edgecolor='black',
            label=label
        )
        for label, color in NODE_COLORS.items()
    ]

    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=10,
        frameon=True
    )

    plt.axis('off')

    plt.tight_layout()

    # -----------------------------------------------------
    # SAVE SEPARATELY
    # -----------------------------------------------------

    png_name = f"case_{idx+1}_knowledge_graph.png"
    pdf_name = f"case_{idx+1}_knowledge_graph.pdf"

    plt.savefig(
        png_name,
        dpi=600,
        bbox_inches='tight'
    )

    plt.savefig(
        pdf_name,
        bbox_inches='tight'
    )

    print(f"Saved: {png_name}")
    print(f"Saved: {pdf_name}")

    plt.close()

print("\nAll knowledge graphs generated successfully!")