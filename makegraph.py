from graphviz import Graph


def make_graph(W, g, cuts, max_weight, name='default'):
    # prevent edge lengths from being 0.
    e = 0.01
    dot = Graph(format='png', engine='neato')
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
                k = W[item.index][tmp.index] / max_weight
                k = 1 - k + e
                dot.attr('edge', len=str(k))
                dot.edge(str(item.index), str(tmp.index))
            q.append(item)

    dot.render('out/%s.gv' % name, view=True)
