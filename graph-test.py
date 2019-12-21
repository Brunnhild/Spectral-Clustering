from graphviz import Graph

dot = Graph(comment='The Round Table', format='png', engine='neato')

dot.attr('node', shape='plaintext', color='red')

dot.node('N3')


dot.node('N1')
dot.node('N2')
dot.attr('edge', len='1')
dot.edge('N1', 'N2')
dot.edge('N2', 'N3')
dot.attr('edge', len='0.5')
dot.edge('N1', 'N3', constraint='false')

print(dot.source)

dot.render('out/round-table.gv', view=True)
