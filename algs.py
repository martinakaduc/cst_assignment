from heapq import heappush, heappop
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function

def dijkstra_path(G, source, target, weight='weight'):
    (length, path) = single_source_dijkstra(G, source, target=target,
                                            weight=weight)
    try:
        return path[target]
    except KeyError:
        raise nx.NetworkXNoPath(
            "node %s not reachable from %s" % (source, target))

def single_source_dijkstra(G, source, target=None, cutoff=None, weight='weight'):
    if source == target:
        return ({source: 0}, {source: [source]})

    if G.is_multigraph():
        get_weight = lambda u, v, data: min(
            eattr.get(weight, 1) for eattr in data.values())
    else:
        get_weight = lambda u, v, data: data.get(weight, 1)

    paths = {source: [source]}  # dictionary of paths
    return _dijkstra(G, source, get_weight, paths=paths, cutoff=cutoff,
                     target=target)


def _dijkstra(G, source, get_weight, pred=None, paths=None, cutoff=None,
              target=None):
    G_succ = G.succ if G.is_directed() else G.adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {source: 0}
    c = count()
    fringe = []  # use heapq with (distance,label) tuples
    push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break

        for u, e in G_succ[v].items():
            cost = get_weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + get_weight(v, u, e)
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    if paths is not None:
        return (dist, paths)
    if pred is not None:
        return (pred, dist)
    return dist



def floyd_warshall_predecessor_and_distance(G, weight='weight'):
    from collections import defaultdict
    # dictionary-of-dictionaries representation for dist and pred
    # use some defaultdict magick here
    # for dist the default is the floating point inf value
    dist = defaultdict(lambda : defaultdict(lambda: float('inf')))
    for u in G:
        dist[u][u] = 0
    pred = defaultdict(dict)
    # initialize path distance dictionary to be the adjacency matrix
    # also set the distance to self to 0 (zero diagonal)
    undirected = not G.is_directed()
    for u,v,d in G.edges(data=True):
        e_weight = d.get(weight, 1.0)
        dist[u][v] = min(e_weight, dist[u][v])
        pred[u][v] = u
        if undirected:
            dist[v][u] = min(e_weight, dist[v][u])
            pred[v][u] = v
    for w in G:
        for u in G:
            for v in G:
                if dist[u][v] > dist[u][w] + dist[w][v]:
                    dist[u][v] = dist[u][w] + dist[w][v]
                    pred[u][v] = pred[w][v]
    return dict(pred)


def floyd_warshall_path(G, source, target, weight='weight'):
    # could make this its own function to reduce memory costs
    pred = floyd_warshall_predecessor_and_distance(G, weight=weight)

    # Get the path from the predecessor dictionary for the source and target
    path = [target]
    while path[-1] != source:
        path.append(pred[source][path[-1]])
    path.reverse() 
    return path


def astar_path(G, source, target, heuristic=None, weight="weight"):
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            ncost = dist + weight(curnode, neighbor, w)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


def k_shortest_paths(G, source, target, k=1, weight='weight'):
    paths = []
    shortest_path_solver = yen(G, source, target, weight=weight)
    for _ in range(k):
        try:
            path = next(shortest_path_solver)
            paths.append(path)
        except:
            break
    return paths


def yen(G, source, target, weight=None):
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target not in G:
        raise nx.NodeNotFound(f"target node {target} not in graph")

    wt = _weight_function(G, weight)

    def length_func(path):
        return sum(
            wt(u, v, G.get_edge_data(u, v)) for (u, v) in zip(path, path[1:])
        )

    shortest_path_func = _bidirectional_dijkstra

    listA = list()
    listB = PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            length, path = shortest_path_func(G, source, target, weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = shortest_path_func(
                        G,
                        root[-1],
                        target,
                        ignore_nodes=ignore_nodes,
                        ignore_edges=ignore_edges,
                        weight=weight,
                    )
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break

class PathBuffer:
    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path



def _bidirectional_dijkstra(
    G, source, target, weight="weight", ignore_nodes=None, ignore_edges=None
):
    if ignore_nodes and (source in ignore_nodes or target in ignore_nodes):
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} not in graph")
        return (0, [source])

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors

    # support optional nodes filter
    if ignore_nodes:

        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    # support optional edges filter
    if ignore_edges:
        if G.is_directed():

            def filter_pred_iter(pred_iter):
                def iterate(v):
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w

                return iterate

            def filter_succ_iter(succ_iter):
                def iterate(v):
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)

        else:

            def filter_iter(nodes):
                def iterate(v):
                    for w in nodes(v):
                        if (v, w) not in ignore_edges and (w, v) not in ignore_edges:
                            yield w

                return iterate

            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{source: [source]}, {target: [target]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{source: 0}, {target: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        wt = _weight_function(G, weight)
        for w in neighs[dir](v):
            if dir == 0:  # forward
                minweight = wt(v, w, G.get_edge_data(v, w))
                vwLength = dists[dir][v] + minweight
            else:  # back, must remember to change v,w->w,v
                minweight = wt(w, v, G.get_edge_data(w, v))
                vwLength = dists[dir][v] + minweight

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")