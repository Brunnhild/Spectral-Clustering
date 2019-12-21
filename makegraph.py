from graphviz import Graph


def make_graph(W, g, cuts):
    dot = Graph(format='png')
    for node in g:
        if node.parent == None:
            dot.attr('node', shape='rectangle', color='red')
        dot.node(str(node.index), str.format('Node %d' % node.index))
        dot.attr('node', shape='', color='')

    q = [g[0]]
    while len(q) > 0:
        tmp = q.pop()
        for item in tmp.children:
            if item.parent != None:
                dot.edge(str(item.index), str(tmp.index))
            q.append(item)

    dot.render('out/tree.gv', view=True)