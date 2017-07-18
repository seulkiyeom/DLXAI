import nn_patterns

output_layer = creat_a_lasagne_network()
pattern = load_pattern()

explainer = nn_patterns.create_explainer("patternnet", output_layer, patterns=patterns)