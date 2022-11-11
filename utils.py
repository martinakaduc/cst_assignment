import time
import random
import osmnx as ox
import folium as fl
from algs import dijkstra_path, floyd_warshall_path, astar_path, k_shortest_paths

folium_map = fl.Map(location=[10.762622, 106.660172], zoom_start=13)
folium_map.add_child(fl.ClickForMarker(popup=None))
# map.add_child(fl.LatLngPopup())

list_points = []
flag = [False]
new_map = [None]
n_nodes = [None]
n_edges = [None]
distance_route = [None]

edge_attr = "length"
# edge_attr = "travel_time"

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def get_pos(point):
    return point['lat'], point['lng']

def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def distance_real(point1, point2):
    return ox.distance.great_circle_vec(point1[0], point1[1], point2[0], point2[1])

def length_of_route(route):
    """
    Calculate the length of a route using the great circle distance between the nodes.
    """
    length = 0
    for i in range(len(route)-1):
        length += distance_real(route[i], route[i+1])
    return length

def get_node_by_id(G, node_id):
    return G.nodes[node_id]

def get_nodes_by_ids(G, list_ids):
    return list(map(lambda x: get_node_by_id(G, x), list_ids))

def find_shortest_path(alg, fl_map, list_points, k=None):
    # G = ox.graph_from_bbox(north=max(list_points[0][0], list_points[1][0]), 
    #                     south=min(list_points[0][0], list_points[1][0]), 
    #                     east=max(list_points[0][1], list_points[1][1]), 
    #                     west=min(list_points[0][1], list_points[1][1]), 
    #                     truncate_by_edge=True,
    #                     network_type='drive')

    if alg == "k_shortest_paths":
        return find_k_shortest_path(fl_map, list_points, k)
    
    mean_point = ((list_points[0][0] + list_points[1][0])/2, (list_points[0][1] + list_points[1][1])/2)
    radius = distance_real(list_points[0], list_points[1]) * 1.5
    G = ox.graph_from_point(mean_point, dist=radius//2, 
                            truncate_by_edge=True, network_type='drive')

    # G = ox.speed.add_edge_speeds(G, fallback=60)
    # G = ox.speed.add_edge_travel_times(G)

    orig = ox.distance.nearest_nodes(G, X=list_points[0][1], Y=list_points[0][0])
    dest = ox.distance.nearest_nodes(G, X=list_points[1][1], Y=list_points[1][0])

    start_time = time.time()
    if alg == "dijkstra":
        route = dijkstra_path(G, orig, dest, edge_attr)
    elif alg == "floyd_warshall":
        route = floyd_warshall_path(G, orig, dest, edge_attr)
    elif alg == "astar":
        def heuristic_dist(x ,y):
            x_coord = get_node_by_id(G, x)
            y_coord = get_node_by_id(G, y)
            return distance_real((x_coord['y'], x_coord['x']), (y_coord['y'], y_coord['x']))

        route = astar_path(G, orig, dest, heuristic=heuristic_dist, weight=edge_attr)
    else:
        route = []
    end_time = time.time()

    print("Route", route)
    print("Time", end_time - start_time)

    m1 = ox.plot_graph_folium(G, graph_map=fl_map, popup_attribute=edge_attr, weight=2, color="#8b0000")
    if len(route) > 0:
        new_map = ox.plot_route_folium(G, route, route_map=m1, popup_attribute=edge_attr, edge_width=6, weight=7)
    new_map.add_child(fl.Marker(location=list_points[0], popup=None))
    new_map.add_child(fl.Marker(location=list_points[1], popup=None))

    list_route_nodes = get_nodes_by_ids(G, list(route))
    mapped_latlong = list(map(lambda x: (x['y'], x['x']), list_route_nodes))
    
    return new_map, G.number_of_nodes(), G.number_of_edges(), length_of_route(mapped_latlong)

def find_k_shortest_path(fl_map, list_points, k):
    mean_point = ((list_points[0][0] + list_points[1][0])/2, (list_points[0][1] + list_points[1][1])/2)
    radius = distance_real(list_points[0], list_points[1]) * 1.5
    G = ox.graph_from_point(mean_point, dist=radius//2, 
                            truncate_by_edge=True, network_type='drive')

    orig = ox.distance.nearest_nodes(G, X=list_points[0][1], Y=list_points[0][0])
    dest = ox.distance.nearest_nodes(G, X=list_points[1][1], Y=list_points[1][0])

    start_time = time.time()
    routes = k_shortest_paths(G, orig, dest, k, edge_attr)
    end_time = time.time()

    route_lengths = []
    
    print("Route", routes)
    print("Time", end_time - start_time)

    m1 = ox.plot_graph_folium(G, graph_map=fl_map, popup_attribute=edge_attr, weight=2, color="#8b0000")
    list_colors = random.choices(colors, k=k)
    new_map = m1
    
    for i, route in enumerate(routes):
        new_map = ox.plot_route_folium(G, route, route_map=new_map, popup_attribute=edge_attr, 
                                        color=list_colors[i], edge_width=6, weight=7)
        list_route_nodes = get_nodes_by_ids(G, list(route))
        mapped_latlong = list(map(lambda x: (x['y'], x['x']), list_route_nodes))
        route_lengths.append(length_of_route(mapped_latlong))

    new_map.add_child(fl.Marker(location=list_points[0], popup=None))
    new_map.add_child(fl.Marker(location=list_points[1], popup=None))
    
    return new_map, G.number_of_nodes(), G.number_of_edges(), route_lengths