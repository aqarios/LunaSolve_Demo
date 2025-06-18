import folium
import matplotlib.pyplot as plt
import networkx as nx
import math
from typing import Dict, List, Tuple, Union


def plot_graph_on_map(G: nx.Graph,
                      cities: Dict[str, Tuple[float, float]]) -> folium.Map:
    """Plot a graph visualization on an interactive map showing all city connections.

    Creates an interactive Folium map centered on Germany that displays all cities
    as markers and all graph edges as blue lines. This provides a visual representation
    of the complete graph structure before optimization.

    Args:
        G: NetworkX graph object containing the cities as nodes and connections as edges.
            The node names should match the keys in the cities dictionary.
        cities: Dictionary mapping city names to their geographic coordinates as
            (latitude, longitude) tuples in decimal degrees format.

    Returns:
        Interactive Folium map object with city markers and edge connections that
        can be displayed in Jupyter notebooks or saved as HTML.

    Example:
        >>> import networkx as nx
        >>> G = nx.complete_graph(['Munich', 'Berlin', 'Hamburg'])
        >>> cities = {
        ...     'Munich': (48.1351, 11.5820),
        ...     'Berlin': (52.5200, 13.4050),
        ...     'Hamburg': (53.5511, 9.9937)
        ... }
        >>> map_viz = plot_graph_on_map(G, cities)
        >>> map_viz.save('graph_visualization.html')
    """
    # Create the plot centered on Germany
    map = folium.Map(location=[51.1657, 10.4515], zoom_start=7)

    # Add city markers
    for city, (lat, lon) in cities.items():
        folium.Marker([lat, lon], popup=city).add_to(map)

    # Add graph edges as blue lines
    for city1, city2 in G.edges():
        lat1, lon1 = cities[city1]
        lat2, lon2 = cities[city2]

        # Add the line connection
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color='blue',
            weight=3,
            opacity=0.7,
        ).add_to(map)

    return map


def plot_solution_tour(
        G: nx.Graph,
        cities: Dict[str, Tuple[float, float]],
        tour: List[str],
        distance_matrix: Union[Dict[str, Dict[str, float]], 'pd.DataFrame']
) -> folium.Map:
    """Plot TSP solution tour on an interactive map with detailed visualization.

    Creates a comprehensive visualization of the TSP solution showing:
    - Complete graph structure in light gray
    - Optimal tour path highlighted in red with directional arrows
    - Numbered city markers indicating tour sequence
    - City name labels for easy identification
    - Information popup on hover/click with distances
    - Legend with total tour distance and explanation

    Args:
        G: NetworkX graph object containing all possible city connections.
            Used to display the complete graph structure in the background.
        cities: Dictionary mapping city names to coordinates as (latitude, longitude)
            tuples in decimal degrees format.
        tour: Ordered list of city names representing the TSP solution tour.
            Should start and end with the same city to represent a closed tour.
        distance_matrix: Distance matrix as either a nested dictionary
            (distance_matrix[city1][city2]) or pandas DataFrame with city names
            as indices/columns. Contains distances between all city pairs.

    Returns:
        Interactive Folium map object with complete TSP solution visualization
        including tour path, city markers, arrows, and informational legend.

    Note:
        The function assumes the tour starts and ends with the same city.
        Arrow directions indicate the tour progression between consecutive cities.
        Distance values are displayed in kilometers and rounded to whole numbers.

    Example:
        >>> tour = ['Munich', 'Berlin', 'Hamburg', 'Munich']
        >>> distance_matrix = {
        ...     'Munich': {'Berlin': 504, 'Hamburg': 612},
        ...     'Berlin': {'Munich': 504, 'Hamburg': 255},
        ...     'Hamburg': {'Munich': 612, 'Berlin': 255}
        ... }
        >>> solution_map = plot_solution_tour(G, cities, tour, distance_matrix)
        >>> solution_map.save('tsp_solution.html')
    """
    # Create the Folium map centered on Germany
    map = folium.Map(location=[51.1657, 10.4515], zoom_start=7)

    # First, add all edges in light gray (to show the complete graph)
    for city1, city2 in G.edges():
        lat1, lon1 = cities[city1]
        lat2, lon2 = cities[city2]

        # Add the line connection
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color='lightgray',
            weight=1,
            opacity=0.3,
        ).add_to(map)

    # Now add the optimal tour path with arrows
    for i in range(len(tour) - 1):
        city1 = tour[i]
        city2 = tour[i + 1]
        lat1, lon1 = cities[city1]
        lat2, lon2 = cities[city2]

        # Calculate distance for popup (handle both dict and DataFrame formats)
        if hasattr(distance_matrix, 'loc'):  # pandas DataFrame
            dist = distance_matrix.loc[city1, city2]
        else:  # nested dictionary
            dist = distance_matrix[city1][city2]

        # Add the tour edge
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color='red',
            weight=5,
            opacity=0.8,
            popup=f"{city1} → {city2}: {dist:.0f} km"
        ).add_to(map)

        # Add arrow marker at the midpoint
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2

        # Calculate arrow angle (corrected for geographic coordinates)
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Calculate angle in degrees (0° is North, 90° is East)
        angle = math.degrees(math.atan2(dlon, dlat))

        # Create a custom arrow using HTML/CSS
        arrow_icon = folium.DivIcon(
            html=f'''
            <div style="position: relative; width: 40px; height: 40px;">
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%) rotate({angle}deg);
                    width: 0;
                    height: 0;
                    border-left: 8px solid transparent;
                    border-right: 8px solid transparent;
                    border-bottom: 25px solid darkred;
                    filter: drop-shadow(0px 0px 2px white);
                "></div>
            </div>
            ''',
            icon_size=(40, 40),
            icon_anchor=(20, 20)
        )

        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=arrow_icon
        ).add_to(map)

    # Add city markers with tour position
    for idx, city in enumerate(tour[:-1]):  # Exclude the last duplicate
        lat, lon = cities[city]

        # Create numbered marker
        number_icon = folium.DivIcon(
            html=f'''
            <div style="font-size: 12pt; color: white; font-weight: bold;
                        text-align: center; vertical-align: middle;
                        background-color: red; border-radius: 50%;
                        width: 30px; height: 30px; line-height: 30px;
                        border: 2px solid darkred;
                        box-shadow: 0px 0px 4px rgba(0,0,0,0.5);">
                {idx + 1}
            </div>
            ''',
            icon_size=(30, 30),
            icon_anchor=(15, 15)
        )

        folium.Marker(
            [lat, lon],
            popup=f"<b>{city}</b><br>Position {idx + 1} in tour",
            icon=number_icon,
            tooltip=f"{idx + 1}. {city}"
        ).add_to(map)

    # Add city name labels
    for city, (lat, lon) in cities.items():
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'''
                <div style="font-size: 11pt; color: black; font-weight: bold;
                            text-shadow: 2px 2px 4px white, -2px -2px 4px white,
                                         2px -2px 4px white, -2px 2px 4px white;
                            margin-top: -38px; margin-left: -35px; width: 70px;
                            text-align: center;">
                    {city}
                </div>
                ''',
                icon_size=(70, 20),
                icon_anchor=(35, 10)
            )
        ).add_to(map)

    # Calculate total distance and add legend
    total_distance = 0
    for i in range(len(tour) - 1):
        if hasattr(distance_matrix, 'loc'):  # pandas DataFrame
            total_distance += distance_matrix.loc[tour[i], tour[i + 1]]
        else:  # nested dictionary
            total_distance += distance_matrix[tour[i]][tour[i + 1]]

    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 240px; height: 150px;
                background-color: white; z-index: 9999;
                border:2px solid grey; border-radius: 5px;
                font-size: 14px; padding: 10px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.3);">
    <p style="margin: 0; font-weight: bold; font-size: 16px;">TSP Solution</p>
    <p style="margin: 5px 0; font-size: 12px;">Tour: {' → '.join(tour)}</p>
    <p style="margin: 5px 0; font-weight: bold; color: darkgreen;">Total Distance: {total_distance:.0f} km</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0;"><span style="color: red; font-weight: bold;">━━━ ▲</span> Optimal tour</p>
    <p style="margin: 5px 0;"><span style="color: lightgray;">━━━</span> All connections</p>
    </div>
    '''

    map.get_root().html.add_child(folium.Element(legend_html))

    return map