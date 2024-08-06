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

    for node in reversed(nodes):
        for parent in node._context.parents:
            dot.edge(parent.__repr__(), node.__repr__(), label=node._context.__class__.__name__)

    ## print the DOT source that will be put in the file
    # print(dot.source)

    ## this will save and render the graph
    dot.render(directory="/tmp", view=True)

    return dot
