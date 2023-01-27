import warnings

import graphviz
import neat

import copy
import os
#%%
def get_pruned_copy(genome, config):
    used_node_genes, used_connection_genes = get_pruned_genes(genome.nodes, genome.connections,
                                                              config.input_keys, config.output_keys)
    new_genome = neat.DefaultGenome(None)
    new_genome.nodes = used_node_genes
    new_genome.connections = used_connection_genes
    return new_genome


#%%
def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=False, show_unused=False,
             node_colors=None, fmt='png', title = None):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    
    if not show_unused:
        def add_children(parents, conns):
            children = set()
            for parent in parents:
                for cg in conns:
                    if cg.enabled:
                        i, o = cg.key
                        if o == parent:
                            children.add(i)
            if len(children) == 0:
                return set()
            else:
                grand_children = add_children(children, conns)
                
                return children.union(grand_children)
        
        used_nodes = add_children(config.genome_config.output_keys, genome.connections.values())
        print(used_nodes)
    else:
        used_nodes = set(genome.nodes.keys()).union(config.genome_config.input_keys)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        if k in used_nodes:
            name = node_names.get(k, str(k))
            #print(f"input: adding {name}")
            input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
            dot.node(name, _attributes=input_attrs)
            
        
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        used_nodes.add(k)
        node_names[k] = f"{node_names[k]}\nBias {genome.nodes[k].bias:.2f}"
        name = node_names.get(k, str(k))
        print(f"output: adding {name}")

        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue'), "width":"0.2", "height":"0.2"}

        dot.node(name, _attributes=node_attrs)
        
        
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        
        node_names[n] = f"Hidden node\nBias {genome.nodes[n].bias:.2f}"
        name = node_names.get(n, str(n))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white'), "width":"0.2", "height":"0.2"}
        dot.node(name, _attributes=node_attrs)
        print(f"nodes: adding {str(n)}")


    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            if show_unused or (input in used_nodes and output in used_nodes):
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width, "label": f"{cg.weight:.2f}"})
            
    if title:
        dot.attr(label=title)
    dot.render(filename, view=view)
    os.remove(filename)
    return dot


#%%
pop = neat.Checkpointer.read_checkpoint("checkpoints/Robobo Experiment 2023-01-25 19;37 - Generation 50")

best_genome = max(pop.population, key = lambda x: pop.population.get(x).fitness)
genome = pop.population[best_genome]

z=draw_net(pop.config, genome,view=True, filename= "image", show_unused = False, title = "Visualisation of best genome",
           node_names = {0:"Left", 1:"Right",
                         -1:"irs back-right", -2:"irs back-middle", -3:"irs back-left", -4:"irs side-right", -5:"irs front-right",
                         -6:"irs front-middle", -7:"irs front-left", -8:"irs side-left", -9:"cam top-left", -10:"cam top-middle",
                         -11:"cam top-right", -12:"cam middle-left", -13:"cam middle", -14:"cam middle-right",
                         -15:"cam bottom-left", -16:"cam bottom-middle", -17:"cam bottom-right"})