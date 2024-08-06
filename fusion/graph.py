from graphviz import Digraph


def topological_sort(self):
    def _topological_sort(node, node_visited, graph_sorted):
        if node not in node_visited:
            if getattr(node, "_context", None):
                node_visited.add(node)
                for parent in node._context.parents:
                    _topological_sort(parent, node_visited, graph_sorted)
                graph_sorted.append(node)
        return graph_sorted

    return _topological_sort(self, set(), [])


def draw_graph(root, filename, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes = topological_sort(root)
    dot = Digraph(filename=filename, format=format, graph_attr={"rankdir": rankdir})  # , node_attr={'rankdir': 'TB'})

    for node in nodes:
        ## no grad for now
        dot.node(name=str(id(node)), label="{ data %.4f | gradient %.4f }" % (node.ndata, node.gradient), shape="record")
        # dot.node(name=str(id(n)), label = "{ %s | data %.4f | gradient %.4f }" % (n.label, n.data, n.gradient), shape='record')
        # if node._operation:
        #     dot.node(name=str(id(node)) + node._operation, label=node._operation)
        #     dot.edge(str(id(node)) + node._operation, str(id(node)))

    # for n1, n2 in edges:
    #     dot.edge(str(id(n1)), str(id(n2)) + n2._operation)

    ## print the DOT source that will be put in the file
    # print(dot.source)

    ## this will save and render the graph
    dot.render(directory="", view=True)

    return dot
