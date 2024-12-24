#%%
import dgl
import networkx as nx
from pyvis.network import Network

#%%

# import dgl.data
# import matplotlib.pyplot as plt
# import networkx as nx

# dataset = dgl.data.CoraGraphDataset()
# g = dataset[0]
# options = {
#     'node_color': 'black',
#     'node_size': 20,
#     'width': 1,
# }
# G = dgl.to_networkx(g)
# plt.figure(figsize=[15,7])
# nx.draw(G, **options)

from pyvis.network import Network

g = Network()
g.add_node(0)
g.add_node(1)
g.add_edge(0, 1)
g.show("data/basic.html")


#%%


# Define number of nodes to show in visualization
nb_nodes_plot = 100

# Step 1. Load the cora dataset and slice it using DGL
dataset = dgl.data.CoraGraphDataset()
g_dgl = dataset[0].subgraph(list(range(nb_nodes_plot)))

# Step 2. Convert the DGLGraph to a NetworkX graph
g_netx = nx.Graph(g_dgl.to_networkx())
assert nb_nodes_plot == g_netx.number_of_nodes()                           # Quickly checks the conversion

# Step 3. Get and assign colors to networkX graph as node attributes 
classes = g_dgl.ndata['label'].numpy()
c_dict = {4:'red', 3:'black'}
colors = {i:c_dict.get(classes[i], 'blue') for i in range(nb_nodes_plot)}  # Build the colors from classes
nx.set_node_attributes(g_netx, colors, name="color")                       # Add the colors as node attributes

# Step 4. Get and assign sizes proportional to the classes found in DGL
sizes = {i:int(classes[i])+1 for i in range(nb_nodes_plot)}
nx.set_node_attributes(g_netx, sizes, name="size")

# Step 5. Get and assign pyvis labels for elegant plotting
labels = {i:str(i) for i in range(nb_nodes_plot)}
nx.set_node_attributes(g_netx, labels, name="label")

# Step 6. Remap the node ids to strings to avoid error with PyVis
g_netx = nx.relabel_nodes(g_netx, labels)                                # 'Relabeling' the nodes ids

# Step 7. Plot the resulting netwrokX graph using PyVis
g_pyvis = Network(height=1500, width=1500, notebook=False)
g_pyvis.from_nx(g_netx, node_size_transf=lambda n:5*n)
g_pyvis.show_buttons(filter_=['nodes'])                                  # Option to control visualization of nodes
g_pyvis.show('data/cora.html')



#%%




# %%
