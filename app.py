import copy
import folium as fl
import streamlit as st
from streamlit_folium import st_folium
from utils import folium_map
from utils import list_points, new_map, n_nodes, n_edges, distance_route, flag, edge_attr
from utils import get_pos, distance, find_shortest_path, find_k_shortest_path

if __name__ == '__main__':
    st.title("Find shortest path")
    alg = st.selectbox('Algorithm', ['dijkstra', 'floyd_warshall', 'astar', 'k_shortest_path'])

    map = st_folium(folium_map, width=1000, height=500)

    # print(map)
    if map["last_clicked"]:
        data = get_pos(map['last_clicked'])
        if data is not None:
            if len(list_points) > 0:
                if distance(list_points[-1], data) > 0.0001:
                    list_points.append(data)
            else:
                list_points.append(data)
        
        if len(list_points) > 2:
            list_points.pop(0)
            list_points.pop(0)
            new_map = [None]
            n_nodes = [None]
            n_edges = [None]
            distance_route = [None]
            flag[0] = False

        if len(list_points) == 2 and not flag[0]:
            # with st.spinner(text='Finding Shortest Path...'):
            new_map[0], n_nodes[0], n_edges[0], distance_route[0] = find_shortest_path(alg, folium_map, list_points)
            flag[0] = True
            st.success('Success!')

            if new_map[0] is not None:
                map = st_folium(new_map[0], width=1000, height=500)

    if new_map[0] is not None:
        st.header("Shortest Path")

    for i, point in enumerate(list_points):
        st.info("Point {}: {}".format(i+1, point))

    if n_nodes[0] is not None and n_edges[0] is not None:
        st.info("Number of nodes: {}".format(n_nodes[0]))
        st.info("Number of edges: {}".format(n_edges[0]))
        if edge_attr == "length":
            st.info("Length of route: {} km".format(distance_route[0]))
        else:
            st.info("Travel time of route: {} minutes".format(distance_route[0]))
    
