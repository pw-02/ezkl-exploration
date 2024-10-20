"""
@file
@brief Helpers to split an ONNX models.
"""
# pylint: disable=R0912
import textwrap
import numpy
from onnx import (  # pylint: disable=E0611
    ModelProto, NodeProto, shape_inference, TensorProto, ValueInfoProto)
from onnx.helper import make_graph, make_model, make_tensor_value_info


class OnnxSegment:
    """
    A segments of an onnx graph assuming
    it is the concatenation of all segments.

    :param parent: an instance of OnnxSplitting
    :param begin: names of the first extremity,
        None for the inputs of the main graph
    :param end: names of the second extremity,
        None for the outputs of the main graph
    :param size: total size of the segment
    :param involved: list of result names involved in this segment
    :param nodes: involved nodes, list of tuple `(int, NodeProto)`
    :param shape_results: name of the shape results, they are not included
        in the segment but should be added to build the final ONNX
    """

    def __init__(self, parent, begin, end, size=0, involved=None, nodes=None,
                 shape_results=None):
        if begin is not None and not isinstance(begin, str):
            raise ValueError(f"begin={begin!r} must be a string or None.")
        if end is not None and not isinstance(end, str):
            raise ValueError(f"end={end!r} must be a string or None.")
        if begin is None and end is None:
            raise ValueError(
                "A segment cannot contain this whole model, "
                "begin and end are both None.")
        if nodes is not None and len(nodes) == 0:
            raise ValueError(
                f"A segment has no node, begin={begin!r}, "
                f"end={end!r}, involved={involved!r}.")
        self.parent = parent
        self.begin = begin
        self.end = end
        self.involved = involved
        self.size = size
        self.nodes = nodes
        self.shape_results = shape_results

    def __repr__(self):
        rows = []
        if self.involved:
            rows.append(f"involved={self.involved!r}")
        if self.shape_results:
            rows.append(f"involved={self.shape_results!r}")
        text = ", ".join(rows)
        if text:
            text = ", " + text
        return f"{self.__class__.__name__}(...,\n    " + "\n".join(
            textwrap.wrap(
                f"{self.begin!r}, {self.end!r}, size={self.size!r}{text}",
                subsequent_indent="    "))


class OnnxSplitting:
    """
    The final goal is to split an onnx model into
    equivalent pieces.

    :param onnx_model: onnx_model
    :param verbose: displays information during the split
    :param doc_string: fills node doc_string to add information about
        the split, no copy is done so it modifies the input nodes as well
    :param fLOG: logging function
    """

    def __init__(self, onnx_model, verbose=0, doc_string=False, fLOG=None):
        self.onnx_model = onnx_model
        self.verbose = verbose
        self.doc_string = doc_string
        self.fLOG = fLOG or print
        self._init()

    @staticmethod
    def _key(idn, node):
        return f"{node.op_type}-{node.name}-{idn}"

    def _init(self):
        onnx_model = self.onnx_model
        if not isinstance(onnx_model, ModelProto):
            raise TypeError(
                f"onnx_model must a ModelProto not a {type(onnx_model)}.")
        node_list = list(enumerate(onnx_model.graph.node))

        # inputs, initializers, constants
        inputs = {}
        initializers = {}
        sparse_initializers = {}
        inputs_only = {}
        constants = {}
        for i in onnx_model.graph.input:
            inputs[i.name] = i
            inputs_only[i.name] = i
        for i in onnx_model.graph.initializer:
            inputs[i.name] = i
            initializers[i.name] = i
        for i in onnx_model.graph.sparse_initializer:
            inputs[i.name] = i
            sparse_initializers[i.name] = i
        for i, n in enumerate(onnx_model.graph.node):
            if n.op_type != "Constant":
                continue
            constants[n.output[0]] = n
        self.inputs = inputs
        self.inputs_only = inputs_only
        self.sparse_initializers = sparse_initializers
        self.initializers = initializers
        self.constants = constants

        # sizes
        sizes = {}
        for init in onnx_model.graph.initializer:
            sizes[init.name] = len(init.SerializeToString())
        for init in onnx_model.graph.sparse_initializer:
            sizes[init.name] = len(init.SerializeToString())
        for idn, node in node_list:
            if node.op_type == "Constant":
                sizes[node.output[0]] = len(node.SerializeToString())
            else:
                sizes[self._key(idn, node)] = len(node.SerializeToString())
        self.sizes = sizes

        # only working for standard domain (supporting shape inference)
        for node in onnx_model.graph.node:
            if node.domain not in {'', 'ai.onnx', 'ai.onnx.ml'}:
                raise NotImplementedError(
                    f"Node {node.op_type!r} from domain {node.domain!r} "
                    f"is not supported yet.")

        # tag shape nodes
        if self.verbose > 0:
            self.fLOG(
                f"[OnnxSplitting._init] mark shape nodes among "
                f"{len(node_list)} nodes.")
        self.node_list_debug = node_list
        shape_obj = self._make_shape_nodes(node_list)
        self.shape_obj = shape_obj
        self.shape_results = set(k[1] for k in shape_obj if k[0] == 0)
        self.shape_nodes = set(k[1] for k in shape_obj if k[0] == 1)
        if self.verbose > 0:
            self.fLOG(f"[OnnxSplitting._init] #shape_results = "
                      f"{len(self.shape_results)}:"
                      f"{list(sorted(self.shape_results))[:5]}")
            self.fLOG(f"[OnnxSplitting._init] #shape_nodes = "
                      f"{len(self.shape_nodes)}:"
                      f"{list(sorted(self.shape_nodes))[:5]}")

        # retrieve all related nodes for a shape result
        self.shape_results_nodes = self._contributing_make_shape_nodes(
            node_list, self.shape_nodes, self.shape_results)
        if self.verbose > 2:
            for ii, (k, v) in sorted(enumerate(
                    self.shape_results_nodes.items())):
                txt = "/".join(v[:5])
                self.fLOG(f"[OnnxSplitting._init]   {ii}:shaperesult[{k}]:"
                          f"{len(v)} - {txt}")

        # cut points: results breaking the connexity of the graph
        if self.verbose > 0:
            self.fLOG(
                f"[OnnxSplitting._init] look for cutting points in "
                f"{len(node_list)} nodes.")

        self.cutting_points = self._get_cutting_points(node_list)
        if len(self.cutting_points) == 0:
            msg = self.display_shape_node_result(
                node_list, self.shape_nodes, self.shape_results)
            raise RuntimeError(
                f"No cutting point, the model with "
                f"{len(node_list)} nodes cannot be split.\n{msg}")

        if self.verbose:
            self.fLOG(
                f"[OnnxSplitting._init] #cuttings-points:"
                f"{len(self.cutting_points)}")

        # segments
        if self.verbose >= 10:
            import tqdm  # pylint: disable=C0415
            loop = tqdm.tqdm(range(len(self.cutting_points)))
        else:
            loop = range(len(self.cutting_points))
        segments = []
        for i in loop:  # pylint: disable=C0200
            segments.append(
                self._make_segment(
                    None if i == 0 else self.cutting_points[i - 1],
                    self.cutting_points[i]))
        segments.append(self._make_segment(self.cutting_points[-1], None))
        self.segments = segments
        if self.verbose > 0:
            self.fLOG(f"[OnnxSplitting._init] #segments:{len(sizes)}")
            self.fLOG("[OnnxSplitting._init] run shape_inference")
        self.shapes = shape_inference.infer_shapes(onnx_model)

        if self.verbose > 0:
            sizes = [seg.size for seg in self.segments]
            self.fLOG(f"[OnnxSplitting._init] #segments:{len(sizes)}, "
                      f"min,avg,max-size=[{min(sizes)}, "
                      f"{sum(sizes) / len(sizes)}, {max(sizes)}]")

    def display_shape_node_result(self, nodes=None, shape_nodes=None,
                                  shape_results=None):
        """
        Displays a message sum
        """
        if nodes is None:
            nodes = self.node_list_debug
        if shape_nodes is None:
            shape_nodes = self.shape_nodes
        if shape_results is None:
            shape_results = self.shape_results
        rows = []
        for idn, node in nodes:
            key = self._key(idn, node)
            inps = []
            for inp in node.input:
                t = "|S" if inp in shape_results else ""
                inps.append(f"{inp}{t}")
            sinps = ", ".join(inps)
            outs = []
            for out in node.output:
                t = "|S" if out in shape_results else ""
                outs.append(f"{out}{t}")
            souts = ", ".join(outs)

            if key in shape_nodes:
                inps = ", ".join(node.input)
                outs = ", ".join(node.output)
                rows.append(f"{node.op_type}|NS({sinps}) -> {souts}")
            else:
                rows.append(f"{node.op_type}({sinps}) -> {souts}")
        return "\n".join(rows)

    @staticmethod
    def _forward_propagate_shape(key, forward_edges):
        """
        From a shape node, walk through the graph
        to find all the nodes contributing to a reshaping.
        """
        dist = {(1, key): 0}
        stack = [(1, key)]
        while len(stack) > 0:
            last = stack.pop()
            d = dist[last] + 1
            if last not in forward_edges:
                # an output
                continue
            ed = forward_edges[last]
            for k, node in ed.items():
                if k not in dist:
                    if k[0] == 1:
                        # node
                        if node.op_type in ("Reshape", "ConstantOfShape"):
                            continue
                    dist[k] = d
                    stack.append(k)
        return dist

    @staticmethod
    def _backward_propagate_shape(key, backward_edges):
        """
        From a shape node, walk through the graph
        to find all the nodes contributing to a reshaping.
        """
        dist = {(1, key): 0}
        stack = [(1, key)]
        while len(stack) > 0:
            last = stack.pop()
            d = dist[last] - 1
            if last not in backward_edges:
                # an input
                continue
            ed = backward_edges[last]
            for k, node in ed.items():
                if k[0] == 1:
                    # node
                    if node.op_type in ("Shape", ):
                        continue
                if k not in dist:
                    dist[k] = d
                    stack.append(k)
        return dist

    def _make_shape_nodes(self, node_list):
        """
        *shape nodes* are nodes operating on shapes, they are usually
        small and can be ignored while looking for the cutting points.
        """
        shapeops = []
        forward_edges = {}
        backward_edges = {}
        for idn, node in node_list:
            if node.op_type == "Constant":
                continue
            key = self._key(idn, node)

            # forward
            for i in node.input:
                if (0, i) not in forward_edges:
                    forward_edges[0, i] = {}
                forward_edges[0, i][1, key] = node
            forward_edges[1, key] = {}
            for o in node.output:
                forward_edges[1, key][0, o] = node
            if node.op_type in ('Shape', 'Size'):
                shapeops.append((idn, node))

            # backward
            backward_edges[1, key] = {}
            for i in node.input:
                backward_edges[1, key][0, i] = node
            for o in node.output:
                if (0, o) not in backward_edges:
                    backward_edges[0, o] = {}
                backward_edges[0, o][1, key] = node

        if len(shapeops) == 0:
            return {}

        # There are shapes, let's propagate.
        if self.verbose >= 10:
            import tqdm  # pylint: disable=C0415
            loop = tqdm.tqdm(shapeops)
        else:
            loop = shapeops
        marked = {}
        for idn, shape in loop:
            key = self._key(idn, shape)
            marked[1, key] = [(0, shape)]
            prop = self._forward_propagate_shape(key, forward_edges)
            for k, v in prop.items():
                if k not in marked:
                    marked[k] = []
                marked[k].append((v, shape))
            # backward propagation, a node is taking a shape result as input,
            # what about the other inputs?
            # Concat(shape, result) -> result contributes to the shape
            for fk, fv in prop.items():
                if fv == 0 or fk[0] == 0:
                    # We skip the node Shape.
                    continue
                back_prop = self._backward_propagate_shape(
                    fk[1], backward_edges)
                for k, v in back_prop.items():
                    if k not in marked:
                        marked[k] = []
                    marked[k].append((v, shape))
        return marked

    def _contributing_make_shape_nodes(self, node_list, shape_nodes, shape_results):
        """
        For all results in shape_results, make the list of contributing
        nodes which are needed to build this result.
        """
        key_node = {self._key(idn, node): node
                    for idn, node in node_list
                    if node.op_type != "Constant"}
        backward = {}
        for key, node in key_node.items():
            for o in node.output:
                backward[o] = key
        res = {}
        for name in shape_results:
            if name not in backward:
                # An initializer
                res[name] = []
                continue
            r = []
            stack = [backward[name]]
            while len(stack) > 0:
                proc = stack
                stack = []
                for key in proc:
                    r.append(key)
                    node = key_node[key]
                    for i in node.input:
                        if i in shape_results and i in backward:
                            stack.append(backward[i])
            res[name] = r
        return res

    @staticmethod
    def _connex_components(vertices, adja):
        vert = {v: i for i, v in enumerate(vertices)}
        more = True
        while more:
            more = False
            for k, v in adja.items():
                if v == 0:
                    continue
                a, b = k
                if vert[a] == vert[b]:
                    continue
                more = True
                if vert[a] < vert[b]:
                    vert[b] = vert[a]
                else:
                    vert[a] = vert[b]
        return vert

    @staticmethod
    def is_small(tensor):
        """
        Tells if a tensor is small. In that case, all edges to this
        constant are ignored when looking for cutting points.
        The algorithm assumes it can be duplicated in multiple parts.
        It is usually single float constant or shapes.
        """
        if isinstance(tensor, TensorProto):
            if tensor.HasField("segment"):
                raise ValueError(  # pragma: no cover
                    "Currently not supporting loading segments.")
            if tensor.data_type == TensorProto.UNDEFINED:  # pylint: disable=E1101
                raise TypeError(  # pragma: no cover
                    "The element type in the input tensor is not defined.")
            dims = tensor.dims
        elif isinstance(tensor, ValueInfoProto):
            dim = tensor.type.tensor_type.shape.dim
            dims = [d.dim_value for d in dim]
            if any(map(lambda x: not isinstance(x, int), dims)):
                return False
        elif isinstance(tensor, NodeProto) and tensor.op_type == "Constant":
            dims = None
            for attr in tensor.attribute:
                if attr.name in {'value_int', 'value_float', 'value_string'}:
                    return True
                if attr.name == "value":
                    return OnnxSplitting.is_small(attr.t)
                if attr.name == 'value_ints':
                    dims = [len(attr.ints)]
                elif attr.name == 'value_floats':
                    dims = [len(attr.floats)]
                elif attr.name == 'value_strings':
                    dims = [len(attr.strings)]
            if dims is None:
                raise TypeError(  # pragma: no cover
                    f"Unexpected constant {type(tensor)}, ")
                    # f"attribute={','.join(a.name) for a in tensor.attribute}.")
        else:
            raise TypeError(  # pragma: no cover
                f"Unexpected type {type(tensor)}.")
        total = numpy.prod(dims)
        if total < 32:
            # covers small constants, reshaping...
            return True
        return False

    def _get_cutting_points(self, node_list):
        # let's avoid adding small constant
        small_tensors = {
            i.name: self.is_small(i)
            for i in self.onnx_model.graph.initializer}
        small_tensors.update({
            i.name: self.is_small(i)
            for i in self.onnx_model.graph.sparse_initializer})
        small_tensors.update({
            i.name: self.is_small(i)
            for i in self.onnx_model.graph.input})
        set_small = set(k for k, v in small_tensors.items() if v)
        for idn, node in node_list:
            if len(node.input) == 0 and len(node.SerializeToString()) < 128:
                key = self._key(idn, node)
                set_small.add(key)
                set_small |= set(node.output)
            elif node.op_type == "Constant":
                key = self._key(idn, node)
                small_tensors[key] = False

        # adjacency matrix
        no_cutting = (
            set(small_tensors) |
            set(o.name for o in self.onnx_model.graph.output))
        constant_type = {'Constant', 'ConstantOfShape'}
        adja = {}
        vertices = set()
        ordered_names = []
        for idn, node in node_list:
            if node.op_type == "Constant":
                continue
            key = self._key(idn, node)
            if key in set_small or key in self.shape_nodes:
                continue
            if (node.op_type not in constant_type and
                    len(node.output) == 1 and
                    len(node.input) > 0):
                # only single output can be cutting points
                ordered_names.extend(
                    o for o in node.output if o not in no_cutting)
            vertices.add(key)
            vertices |= set(i for i in node.input
                            if i not in set_small and
                            i not in self.shape_results)
            vertices |= set(o for o in node.output
                            if o not in set_small and
                            o not in self.shape_results)
            for i in node.input:
                if i in set_small or i in self.shape_results:
                    continue
                adja[i, key] = 1
            for o in node.output:
                if o in set_small:
                    continue
                adja[key, o] = 1

        # checking the connexity
        if self.verbose >= 10:
            import tqdm  # pylint: disable=C0415
            loop = tqdm.tqdm(ordered_names)
        else:
            loop = ordered_names
        cutting_points = []
        for name in loop:
            keys = []
            for a, b in adja:
                if b == name:
                    keys.append((a, b))

            # remove the links
            for a, b in keys:
                adja[a, b] = 0

            connex = self._connex_components(vertices, adja)
            connex_id = set(connex.values())
            if len(connex_id) == 2:
                cutting_points.append(name)

            # put back the links
            for a, b in keys:
                adja[a, b] = 1

        return cutting_points

    def _segment_names(self, nodes, names, results, skip_names=None):
        if skip_names is None:
            skip_names = set()
        names = names.copy()
        size = 0
        subset = []
        shape_results = set()
        for idn, node in reversed(nodes):
            if node.op_type == "Constant":
                continue
            if set(node.output) & names:
                size += self.sizes[self._key(idn, node)]
                if len(node.output) == 1 and node.output[0] in skip_names:
                    continue
                subset.append((idn, node))
                no_shape = [i for i in node.input
                            if i not in self.shape_results]
                if len(no_shape) == 1 and no_shape[0] in skip_names:
                    for i in node.input:
                        if i in self.shape_results:
                            shape_results.add(i)
                    continue
                for i in node.input:
                    if i in self.shape_results:
                        shape_results.add(i)
                        continue
                    if i in self.sizes:
                        size += self.sizes[i]
                    names.add(i)
        return subset, shape_results, size, names

    def _make_segment(self, name1, name2):
        nodes = []
        for idn, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "Constant":
                continue
            nodes.append((idn, node))
            if name2 is not None and name2 in node.output:
                break

        if name1 is not None and name1 in self.shape_results:
            raise RuntimeError(  # pragma: no cover
                f"Cannot create a segment with a shape result {name1!r} "
                f"as an input.")
        if name2 is not None and name2 in self.shape_results:
            raise RuntimeError(  # pragma: no cover
                f"Cannot create a segment with a shape result {name2!r} "
                f"as an output.")
        if name2 is None:
            names = set(i.name for i in self.onnx_model.graph.output)
        else:
            names = {name2}

        subset, shape_results, size, names = self._segment_names(
            nodes, names, results={name2}, skip_names={name1})
        subset.sort()  # original order must be kept
        involved = names if name2 is None else names - {name2}
        seg = OnnxSegment(self, begin=name1, end=name2,
                          involved=involved,
                          size=size, nodes=subset,
                          shape_results=shape_results)
        return seg

    def _split_2(self, a, b):
        """
        Splits the segments into two groups of the same size.

        :param a: first segment (included)
        :param b: second segment (excluded)
        :return: split index
        """
        if a >= b - 1:
            raise RuntimeError(f"a={a}, b={b}, unable to split.")
        if a == b - 2:
            return a + 1
        sizes = numpy.array([s.size for s in self.segments[a:b]])
        sizes_for = numpy.cumsum(sizes)
        sizes_bck = numpy.cumsum(sizes[::-1])[::-1].copy()
        diff = numpy.abs(sizes_bck - sizes_for)
        pos = numpy.argmin(diff)
        # pos is the beginning of the interval
        pos += 1
        pos += a
        if pos == a:
            pos = a + 1
        elif pos == b:
            pos = b - 1
        return pos

    def split_segment(self, n_parts=None, cut_points=None):
        """
        Splits the segments into `n_parts` segments

        :param n_parts: number of parts to get
        :param cut_points: uses this particular cut points
        :return: list of segments indices
        """
        if n_parts is None and cut_points is None:
            raise ValueError("n_parts or cut_points must be specified.")
        if n_parts is not None and cut_points is not None:
            raise ValueError("n_parts and cut_points cannot "
                             "be specified at the same time.")
        if cut_points is not None:
            possible = set(self.cutting_points)
            for name in cut_points:
                if name not in possible:
                    text = "\n".join(textwrap.wrap(str(self.cutting_points)))
                    raise ValueError(
                        f"Cut point {name!r} is not considered as a cutting "
                        f"points. Possible canditates:\n{text}")
            memo = {s.begin: i for i, s in enumerate(self.segments)
                    if s.begin is not None}
            extremities = [0]
            for name in cut_points:
                extremities.append(memo[name])
            extremities.append(len(self.segments))
            return extremities

        if self.verbose > 10:
            self.fLOG("[OnnxSplitting.split_segment] cutting points")
            self.fLOG("\n".join(textwrap.wrap(
                ", ".join(map(str, self.cutting_points)))))
        extremities = [0, len(self.segments)]
        n = n_parts
        while n > 1:
            if n % 2 != 0:
                raise NotImplementedError(
                    f"n_parts={n_parts} is not a power of 2.")
            new_ext = [extremities[0]]
            for i in range(1, len(extremities)):
                a, b = extremities[i - 1:i + 1]
                if self.verbose > 1:
                    size = sum(s.size for s in self.segments[a:b])
                    names = self.segments[a].begin, self.segments[b - 1].end
                    self.fLOG(f"[OnnxSplitting.split_segment] "
                              f"split into n={n}, from a={a} to b={b}, "
                              f"size={size}, {names[0]!r} -> {names[1]!r}")
                pos = self._split_2(a, b)
                if self.verbose > 1:
                    size_a = sum(s.size for s in self.segments[a:pos])
                    size_b = sum(s.size for s in self.segments[pos:b])
                    self.fLOG(f"[OnnxSplitting.split_segment] found "
                              f"pos={pos}, size_1={size_a}, "
                              f"size_2={size_b}={size_b/size:1.2f}, "
                              f"split={self.segments[pos].begin!r}")
                new_ext.extend([pos, b])
            extremities = new_ext
            n = n // 2
        return extremities

    def make_onnx(self, extremities):
        """
        Builds onnx subparts based on the segmentation
        defined by extremities.

        :param extremities: example, `[0, 3, 5]`,
            first onnx part contains segments `0:3=[0, 1, 2]`,
            second onnx part contains segments `3:5=[3, 4]`
        :return: list of onnx subgraphs (:epkg:`ModelProto`)
        """
        if self.verbose > 0:
            self.fLOG(
                f"[OnnxSplitting.make_onnx] #parts:{len(extremities) - 1}")
        res = []
        needed_inputs = {}
        for i in range(len(extremities) - 1, 0, -1):
            if self.verbose > 0:
                self.fLOG(f"[OnnxSplitting.make_onnx] part {i}:begin")
            a, b = extremities[i - 1:i + 1]
            onx, extra_inputs = self._make_onnx(
                a, b, i - 1, add_inputs=needed_inputs)
            res.append(onx)
            for name in extra_inputs:
                if name not in needed_inputs:
                    needed_inputs[name] = []
                needed_inputs[name].append(i)
            if self.verbose > 0:
                n_nodes = len(self.onnx_model.graph.node)
                total = sum(s.size for s in self.segments)
                size = sum(self.segments[i].size for i in range(a, b))
                self.fLOG(f"[OnnxSplitting.make_onnx] part {i}:end:"
                          f"#nodes={len(onx.graph.node)}"  # pylint: disable=E1101
                          f"/{n_nodes}, size={size}/{total}={size/total:1.2f}, "
                          f"extra_inputs={extra_inputs}")
        if self.verbose > 0:
            self.fLOG(
                f"[OnnxSplitting.make_onnx] needed_inputs={needed_inputs}")
        return list(reversed(res))

    def _process_shape(self, nodes, shape_results, input_names, output_names):
        if shape_results is None or len(shape_results) == 0:
            return None, None, None

        if self.verbose > 1:
            self.fLOG(
                f"[OnnxSplitting._process_shape] {input_names}->{output_names}:#"
                f"{len(shape_results)}")

        subset, shape_results, _, names = self._segment_names(
            nodes, output_names, results=set(output_names))
        graph_nodes = []
        for idn, node in enumerate(self.onnx_model.graph.node):
            graph_nodes.append((idn, node))

        # gather shape nodes
        subset = []
        shape_results_to_add = set()
        new_inputs = set()
        for shape in shape_results:
            if self.verbose > 2:
                self.fLOG(
                    f"[OnnxSplitting._process_shape]   "
                    f"shape:{shape}")
            subset_shape = []
            keys = set(self.shape_results_nodes[shape])
            for idn, node in graph_nodes:
                key = self._key(idn, node)
                if key in keys:
                    subset_shape.append((idn, node))
            included = set()
            for _, node in subset_shape:
                included |= set(node.output)
            no_addition = False
            if self.verbose > 2:
                self.fLOG(
                    f"[OnnxSplitting._process_shape]     "
                    f"+#nodes:{len(subset_shape)}")
            for idn, node in subset_shape:
                for i in node.input:
                    if i in included or i in names or i in input_names:
                        continue
                    if i in self.inputs:
                        if self.verbose > 3:
                            self.fLOG(
                                f"[OnnxSplitting._process_shape]     "
                                f"+in[{i}]")
                        obj = self.inputs[i]
                        if not self.is_small(obj):
                            raise RuntimeError(
                                f"Input {i!r} is needed to reshape "
                                f"but is not small.")
                        names.add(i)
                        new_inputs.add(i)
                    elif i in self.initializers:
                        obj = self.inputs[i]
                        if not self.is_small(obj):
                            raise RuntimeError(
                                f"Initializer {i!r} is needed to reshape "
                                f"but is not small.")
                        if self.verbose > 3:
                            self.fLOG(
                                f"[OnnxSplitting._process_shape]     "
                                f"+ci[{i}]")
                        names.add(i)
                    elif i in self.constants:
                        obj = self.constants[i]
                        if not self.is_small(obj):
                            raise RuntimeError(
                                f"Constant {i!r} is needed to reshape "
                                f"but is not small.")
                        if self.verbose > 3:
                            self.fLOG(
                                f"[OnnxSplitting._process_shape]     "
                                f"+cs[{i}]")
                        names.add(i)
                        new_inputs.add(i)
                    else:
                        if self.verbose > 3:
                            self.fLOG(
                                "[OnnxSplitting._process_shape]     +copied"
                                f" because missing[{i}]")
                        # one input is not an initializer
                        # we look into copying the shape or
                        no_addition = True
                        break
                if no_addition:
                    break

            if no_addition:
                # the shape comes from a tensor not in this segment,
                # the shape has to be considered as an input
                shape_results_to_add.add(shape)
            else:
                # the shape comes from a tensor in this segment
                subset.extend(subset_shape)
        return subset, shape_results_to_add, new_inputs

    def _make_onnx(self, a, b, index=None, add_inputs=None):
        """
        Builds one onnx subpart including segments from a to b (excluded).

        :param a: index of the first segment (or None for the first one)
        :param b: index of the second segment (or None for the last one)
        :param index:
        :param add_inputs: shape result needed in the next parts
        :return: (ONNX model, extra_inputs),
            extra_inputs is the set of inputs this part requires
        """
        if self.verbose > 1:
            self.fLOG(f"[OnnxSplitting._make_onnx] {a}->{b}")
        if index is None:
            index = a

        # common parts
        value_info = {o.name: o for o in self.onnx_model.graph.output}
        for info in self.shapes.graph.value_info:  # pylint: disable=E1101
            if not info.type.tensor_type.HasField("shape"):
                new_info = make_tensor_value_info(
                    info.name, info.type.tensor_type.elem_type, [],
                    doc_string=info.doc_string)
                value_info[info.name] = new_info
            else:
                value_info[info.name] = info

        segs = self.segments[a:b]
        involved = set()
        shape_results = set()
        for seg in segs:
            involved |= seg.involved
            shape_results |= seg.shape_results

        # initializers
        new_inits = [init for init in self.onnx_model.graph.initializer
                     if init.name in involved]
        new_sp_inits = [init for init in self.onnx_model.graph.sparse_initializer
                        if init.name in involved]
        new_constants = [node for name, node in self.constants.items()
                         if name in involved]

        # nodes
        shape_results = set()
        nodes = [(-i - 1, node) for i, node in enumerate(new_constants)]
        idns = set(n[0] for n in nodes)
        if len(idns) != len(nodes):
            raise RuntimeError(
                f"len(idns)={len(idns)} != len(nodes)={len(nodes)}")
        for iseg, seg in enumerate(segs):
            if self.doc_string:
                label = (f"seg{iseg + a}-size={seg.size}-"
                         f"[{seg.begin or ''},{seg.end or ''}]")
            if seg.shape_results:
                if self.doc_string:
                    label += f"-shape={list(sorted(seg.shape_results))}"
                shape_results |= seg.shape_results
            for idn, node in seg.nodes:
                if self.doc_string:
                    if node.doc_string:
                        node.doc_string = f"{node.doc_string}-{label}"
                    else:
                        node.doc_string = label
                if idn in idns:
                    raise RuntimeError(
                        f"A node idn={idn!r} was already added.")
                idns.add(idn)
                nodes.append((idn, node))
        if len(idns) != len(nodes):
            raise RuntimeError(
                f"len(idns)={len(idns)} != len(nodes)={len(nodes)}")

        # inputs, outputs
        existing_inputs = [i for i in self.onnx_model.graph.input
                           if i.name in involved]
        if a == 0:
            new_inputs = existing_inputs
        else:
            new_inputs = [value_info[segs[0].begin]] + existing_inputs

        if b == len(self.segments):
            new_outputs = [i for i in self.onnx_model.graph.output
                           if i.name in involved]
        else:
            new_outputs = [value_info[segs[-1].end]]

        # process shapes
        input_names = set(o.name for o in new_inputs)
        output_names = set(o.name for o in new_outputs)
        new_nodes, shape_to_add, new_internals = self._process_shape(
            nodes, shape_results, input_names, output_names)
        if len(idns) != len(nodes):
            raise RuntimeError(
                f"len(idns)={len(idns)} != len(nodes)={len(nodes)}")
        if new_internals is not None and len(new_internals) > 0:
            # adding initializer or existing inputs
            for name in new_internals:
                if name in self.constants:
                    new_nodes.append(
                        (-len(new_nodes) - 1, self.constants[name]))
                    continue
                if name in self.inputs_only:
                    new_inputs.append(self.inputs_only[name])
                if name in self.initializers:
                    new_inits.append(self.initializers[name])
                elif name in self.sparse_initializers:
                    new_sp_inits.append(self.sparse_initializers[name])

        extra_inputs = []
        if shape_to_add is not None and len(shape_to_add) > 0:
            # add unresolved shapes as new inputs
            for i in shape_to_add:
                extra_inputs.append(i)
                if i in value_info:
                    new_inputs.append(value_info[i])
                else:
                    # shape information is missing
                    new_inputs.append(make_tensor_value_info(
                        i, TensorProto.INT64, [None],
                        doc_string="shape-shape"))
        if len(idns) != len(nodes):
            raise RuntimeError(
                f"len(idns)={len(idns)} != len(nodes)={len(nodes)}")
        if new_nodes is not None:
            for idn, node in new_nodes:
                if idn in idns:
                    raise RuntimeError(
                        f"A node idn={idn!r} was already added.")
                idns.add(idn)
                nodes.append((idn, node))
        if len(idns) != len(nodes):
            raise RuntimeError(
                f"len(idns)={len(idns)} != len(nodes)={len(nodes)}")
        nodes.sort()

        # extra inputs
        if add_inputs is not None:
            in_this = set(i.name for i in new_inputs)
            for _, n in nodes:
                in_this |= set(n.output)
            for name in add_inputs:
                new_outputs.append(value_info[name])
                if self.verbose > 2:
                    self.fLOG(
                        f"[OnnxSplitting._make_onnx]   +output:{name}")
                if name in in_this:
                    # the shape is produced in this part
                    continue
                # the shape comes from a previous shape
                new_inputs.append(value_info[name])
                extra_inputs.append(name)
                in_this.add(name)

        # verification that all inputs are part of the graph
        names = set(i.name for i in new_inits)
        names |= set(i.name for i in new_sp_inits)
        names |= set(i.name for i in new_inputs)
        names |= set(extra_inputs)
        existing_inits = {i.name: i for i in self.onnx_model.graph.initializer}
        existing_sp_inits = {
            i.name: i for i in self.onnx_model.graph.sparse_initializer}
        existing_constants = {
            n.output[0]: (i, n) for i, n in enumerate(self.onnx_model.graph.node)
            if n.op_type == 'Constant'}
        constant_nodes = []
        for _, node in nodes:
            for i in node.input:
                if i not in names:
                    if i in existing_inits:
                        new_inits.append(existing_inits[i])
                    elif i in existing_sp_inits:
                        new_sp_inits.append(existing_sp_inits[i])
                    elif i in existing_constants:
                        constant_nodes.append(existing_constants)
                    else:
                        extra_inputs.append(i)
                    names.add(i)
                    if self.verbose > 2:
                        self.fLOG(
                            f"[OnnxSplitting._make_onnx]   +input-:{i}")
            names |= set(node.output)

        if constant_nodes:
            nodes.extend(constant_nodes)
            nodes.sort()

        # remove node index
        nodes = [n for _, n in nodes]

        model = self.onnx_model
        graph = make_graph(
            nodes, f"{model.graph.name}-{index}",
            new_inputs, new_outputs,
            new_inits, doc_string=model.graph.doc_string,
            sparse_initializer=new_sp_inits,
            value_info=model.graph.value_info)
        new_model = make_model(graph, opset_imports=model.opset_import)
        new_model.ir_version = model.ir_version
        new_model.producer_name = model.producer_name
        new_model.producer_version = model.producer_version
        new_model.domain = model.domain
        new_model.model_version = model.model_version
        if self.doc_string:
            text = (f"segments[{a}:{b}], from {segs[0].begin or ''} "
                    f"to {segs[-1].end or ''}\n"
                    f"shapes_results={shape_results}")
            new_model.doc_string = f"{model.doc_string}\n----\n{text}"
        else:
            new_model.doc_string = model.doc_string
        return new_model, extra_inputs

def get_all_cut_points(onnx_model, verbose=0):
    spl_onnx = OnnxSplitting(onnx_model, verbose=verbose,
                            doc_string=False, fLOG=print)
    return spl_onnx.cutting_points
    
def split_onnx(onnx_model, n_parts=None, cut_points=None,
               verbose=0, stats=False, doc_string=False,
               fLOG=None):
    """
    Splits an ONNX model into *n_parts* consecutive subgraphs.
    Chained altogether, they are equivalent to the given model.

    :param onnx_model: onnx model
    :param n_parts: number of subgraphs
    :param cut_points: use different cutting points that the ones
        the algorithm can found, it must be in the available set
        of cutting points
    :param verbose: display information related to the split
    :param stats: returns statistics as well, return of the
        function is a tuple
    :param doc_string: fills node doc_string to add information about
        the split, no copy is done so it modifies the input nodes as well
    :param fLOG: logging function
    :return: list of onnx model
    """
    if len(onnx_model.functions) > 0:
        raise NotImplementedError(
            f"The function does not work if the model contains function: ")
            # f"{f.name for f in onnx_model.functions}.")
    if n_parts is not None and not isinstance(n_parts, int):
        raise TypeError(
            f"n_parts must be None or an interger not {type(n_parts)}.")
    if cut_points is not None and not isinstance(cut_points, (list, tuple)):
        raise TypeError(
            f"cut_points must be None or a list not {type(n_parts)}.")
    if verbose > 0:
        (fLOG or print)(
            f"[split_onnx] prepare splitting "
            f"{len(onnx_model.graph.node)} nodes in {n_parts} parts.")
    spl_onnx = OnnxSplitting(onnx_model, verbose=verbose,
                             doc_string=doc_string, fLOG=fLOG or print)
    if n_parts is not None and len(spl_onnx.cutting_points) < n_parts:
        raise RuntimeError(  # pragma: no cover
            f"Unable to split the onnn model, there are less cutting points "
            f"{len(spl_onnx.cutting_points)} than the number of requested "
            f"splits ({n_parts}).")
    if verbose > 0:
        (fLOG or print)(
            f"[split_onnx] starts splitting "
            f"{len(onnx_model.graph.node)} nodes in {n_parts} parts.")
    exts = spl_onnx.split_segment(n_parts, cut_points=cut_points)
    if verbose > 0:
        names = [spl_onnx.segments[i].begin for i in exts[1:-1]]
        (fLOG or print)(f"[split_onnx] splits: {exts}, names={names}")
    res = spl_onnx.make_onnx(exts)
    if stats:
        more = dict(
            split=spl_onnx,
            segments=[dict(size=s.size, nodes=len(s.nodes),
                           involved=s.involved)
                      for s in spl_onnx.segments],
            cutting_points=spl_onnx.cutting_points,
            extremities=exts,
            split_points=[spl_onnx.segments[e].begin for e in exts[1:-1]],
            shape_nodes=spl_onnx.shape_nodes,
            shape_results=spl_onnx.shape_results)
        return res, more
    return res


if __name__ == "__main__":
    import onnx
    onnx_model = onnx.load("examples/onnx/nanoGPT/network.onnx")
    result = split_onnx(onnx_model, n_parts=16, verbose=1)
    pass