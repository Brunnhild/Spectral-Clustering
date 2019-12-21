from graphviz import Graph

dot = Graph(comment='The Round Table', format='png')

dot.attr('node', shape='rectangle', color='red')

dot.node('A', 'King Arthur')

dot.attr('node', shape='', color='')

dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')
dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

print(dot.source)

dot.render('out/round-table.gv', view=True)
