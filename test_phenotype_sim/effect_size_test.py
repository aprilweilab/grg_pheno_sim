def find_ancestors(node, grg):
    # this function recursively collects all ancestors of a given node
    ancestors = []
    parents = grg.get_up_edges(node)
    for parent in parents:
        ancestors.append(parent)
        ancestors.extend(find_ancestors(parent, grg))
    return ancestors


def test_individual_sample(sample, grg, causal_mutation_df):
    ancestors = find_ancestors(sample, grg)

    mutation_effect_sizes = {
        int(row["mutation_id"]): float(row["effect_size"])
        for index, row in causal_mutation_df.iterrows()
    }

    # initialize a dictionary to store the cumulative effect sizes for each node
    node_effect_sizes = {node: 0.0 for node in range(grg.num_nodes)}

    # compute initial effect sizes for each node by summing mutations' effect sizes
    for node in node_effect_sizes.keys():
        mutations = grg.get_mutations_for_node(node)
        if mutations:
            total_effect_size = sum(
                mutation_effect_sizes.get(mutation, 0) for mutation in mutations
            )
            node_effect_sizes[node] = total_effect_size

    for node in node_effect_sizes.keys():
        if node in ancestors:
            node_effect_sizes[sample] += node_effect_sizes[node]

    return node_effect_sizes[sample]


def test_additive_effect_sizes(grg, causal_mutation_df):
    sample_nodes = grg.get_sample_nodes()
    sample_effect_sizes = []

    for sample in sample_nodes:
        sample_effect_size = test_individual_sample(sample, grg, causal_mutation_df)
        sample_effect_sizes.append(sample_effect_size)

    return sample_effect_sizes
