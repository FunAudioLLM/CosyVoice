import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Constant


def layernorm_op17_match(graph) -> gs.PatternMapping:
    # opset17
    pattern_op17 = gs.GraphPattern()
    in_op17_0 = pattern_op17.variable()
    in_op17_1 = pattern_op17.variable()
    in_op17_2 = pattern_op17.variable()
    out_op17_0 = pattern_op17.add(name='node_op17_ln', op='LayerNormalization', inputs=[in_op17_0, in_op17_1, in_op17_2])
    pattern_op17.set_output_tensors([out_op17_0])

    matched_subgraphs_op17 = pattern_op17.match_all(graph)
    return matched_subgraphs_op17, pattern_op17


def layernorm_op17_replace(graph, matched_subgraphs):
    for i, match in enumerate(matched_subgraphs):
        print(f'{i}')

        inputs = []
        inputs.append(match.get("node_op17_ln").inputs[0])
        inputs.append(match.get("node_op17_ln").inputs[1])
        inputs.append(match.get("node_op17_ln").inputs[2])
        print(match.get("node_op17_ln").inputs[1].shape)
        # print(match.get("node_op17_ln").inputs[0].shape)

        outputs = []
        outputs.append(match.get("node_op17_ln").outputs[0])

        custom_op = gs.Node(
            op='CusLnm3d_eps6',
            name='cus_layernorm.' + str(i),
            inputs=inputs,
            outputs=outputs,
            domain='cus_onnx_op'
        )

        graph.nodes.append(custom_op)

        rm_nodes_names = []
        for node_name in match.keys():
            node = match.get(node_name)
            graph.nodes.remove(node)
            rm_nodes_names.append(node.name)

        for input in inputs:
            if not isinstance(input, Constant):
                # input.outputs = [custom_op]
                input.outputs = [x for x in input.outputs if x.name not in rm_nodes_names]

        for output in outputs:
            # output.inputs = [custom_op]
            output.inputs = [x for x in output.inputs if x.name not in rm_nodes_names]

    graph.cleanup()
    return graph


def gs4layernorm(model_path, model_rp_path):
    graph = gs.import_onnx(onnx.load(model_path))
    matched_subgraphs_op17, pattern_op17 = layernorm_op17_match(graph)
    print(len(matched_subgraphs_op17))

    graph = layernorm_op17_replace(graph, matched_subgraphs_op17)
    # export ONNX model
    subgraph_model = gs.export_onnx(graph)
    # change IR version
    subgraph_model.ir_version = 10
    print("cur IR version:", subgraph_model.ir_version)
    onnx.save(subgraph_model, model_rp_path)


def main():
    model_path =    '/weights/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.fp32.onnx'
    model_rp_path = '/weights/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.fp32.cus_ln.onnx'
    gs4layernorm(model_path, model_rp_path)


if __name__ == "__main__":
    main()
    print('finish')
