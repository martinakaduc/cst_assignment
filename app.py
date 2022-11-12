import folium as fl
import streamlit as st
from streamlit_folium import st_folium
from utils import edge_attr
from utils import get_pos, distance, find_shortest_path

if __name__ == '__main__':
    st.title("Find shortest path")
    alg = st.selectbox('Algorithm', ['dijkstra', 'floyd_warshall', 'astar', 'k_shortest_paths'])
    if alg == "k_shortest_paths":
        k = st.slider('Number of paths', min_value=1, max_value=10, value=3)
    else:
        k = None

    if 'folium_map' not in st.session_state or st.button('Reset map'):
        st.session_state.folium_map = fl.Map(location=[10.762622, 106.660172], zoom_start=13)
        st.session_state.folium_map.add_child(fl.ClickForMarker(popup=None))

        st.session_state.list_points = []
        st.session_state.flag = False
        st.session_state.new_map = None
        st.session_state.n_nodes = None
        st.session_state.n_edges = None
        st.session_state.distance_route = None

    map = st_folium(st.session_state.folium_map, width=1000, height=500)

    # print(map)
    if map["last_clicked"]:
        data = get_pos(map['last_clicked'])
        if data is not None:
            if len(st.session_state.list_points) > 0:
                if distance(st.session_state.list_points[-1], data) > 0.0001:
                    st.session_state.list_points.append(data)
            else:
                st.session_state.list_points.append(data)
        
        if len(st.session_state.list_points) > 2:
            st.session_state.list_points.pop(0)
            st.session_state.list_points.pop(0)
            st.session_state.flag = False
            st.session_state.new_map = None
            st.session_state.n_nodes = None
            st.session_state.n_edges = None
            st.session_state.distance_route = None

        if len(st.session_state.list_points) == 2 and not st.session_state.flag:
            # with st.spinner(text='Finding Shortest Path...'):
            st.session_state.new_map, st.session_state.n_nodes, st.session_state.n_edges, \
                st.session_state.distance_route = find_shortest_path(alg, st.session_state.folium_map, \
                    st.session_state.list_points, k=k)
            st.session_state.flag = True
            st.success('Success!')

            if st.session_state.new_map is not None:
                map = st_folium(st.session_state.new_map, width=1000, height=500)

    if st.session_state.new_map is not None:
        st.header("Shortest Path")

    for i, point in enumerate(st.session_state.list_points):
        st.info("Point {}: {}".format(i+1, point))

    if st.session_state.n_nodes is not None and st.session_state.n_edges is not None:
        st.info("Number of nodes: {}".format(st.session_state.n_nodes))
        st.info("Number of edges: {}".format(st.session_state.n_edges))
        if edge_attr == "length":
            if isinstance(st.session_state.distance_route, list):
                list_distance = [d/1000 for d in st.session_state.distance_route]
                st.info("Length of route(s): {} km".format(list_distance))
            else:
                st.info("Length of route(s): {} km".format(st.session_state.distance_route/1000))
        else:
            st.info("Travel time of route(s): {} minutes".format(st.session_state.distance_route))
    
