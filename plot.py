import graphviz
import sys
from graphviz import Digraph
import ast


def plot_genotype(flag, genotype, file_name=None, figure_dir='./network_structure', save_figure=False):
    # Set graph style
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='40', fontname="arial"),
        node_attr=dict(style='rounded,filled', shape='box', align='center', fontsize='50', height='0.8', width='0.8',
                       penwidth='1.5', fontname="arial"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    if flag == 'g':
        input_node_names = ['a', 'z', '[a,z]']
    else:
        input_node_names = ['a', 'f', '[a,f]']
    output_name = 'o'

    # All input nodes
    for input_node_name in input_node_names:
        g.node(input_node_name, fillcolor='mediumaquamarine')

    # Number of inner nodes
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    # All inner nodes
    for i in range(steps - 1):
        g.node(str(i), fillcolor='darkgoldenrod1')

    # Output node
    g.node(output_name, fillcolor='cornflowerblue')

    # Add edges
    # Edge direction: u ---> v
    # Genotype: operation, u
    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j < len(input_node_names):
                u = input_node_names[j]
            else:
                u = str(j - len(input_node_names))

            if i == steps - 1:
                v = output_name
            else:
                v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # Save the figure
    if save_figure:
        g.render(file_name, view=False, directory=figure_dir)

    return g

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)
  cur_genotype_G = list(ast.literal_eval(sys.argv[1]))
  cur_genotype_D = list(ast.literal_eval(sys.argv[2]))
  print('generator', cur_genotype_G)
  print('discriminator', cur_genotype_D)
  # try:
  #   genotype = eval('genotypes.{}'.format(genotype_name))
  # except AttributeError:
  #   print("{} is not specified in genotypes.py".format(genotype_name))
  #   sys.exit(1)

  plot_genotype('g',
                cur_genotype_G,
                file_name='test_G',
                #          '%s_%s_%s' % \
                # (opt.figure_dir, opt.dataset, timestamp),
                save_figure=True
                )
  plot_genotype('d',
                cur_genotype_D,
                file_name='test_D',
                #          '%s_%s_%s' % \
                # (opt.figure_dir, opt.dataset, timestamp),
                save_figure=True
                )
  print('Figure saved.')