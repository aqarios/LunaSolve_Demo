import math
import pandas as pd
from typing import Dict, List, Tuple, Union


def extract_tour_from_solution(
        variable_names: List[str],
        sample_values: List[int],
        start_city: str
) -> List[str]:
    """Extract tour from TSP solution with fixed starting city.

    This function reconstructs the tour path from the binary variables of a
    solved Traveling Salesman Problem (TSP). The starting city is fixed at
    position 0, and other cities are placed according to their assigned positions.

    Args:
        variable_names: List of variable names in format 'x_{city}_{position}'
            where each variable represents whether a city is visited at a
            specific position in the tour.
        sample_values: List of binary values (0 or 1) corresponding to each
            variable, where 1 indicates the variable is active in the solution.
        start_city: Name of the city that serves as the starting point of the tour.
            This city is automatically placed at position 0.

    Returns:
        Complete tour as a list of city names, starting and ending with the
        start_city to represent a closed tour.

    Example:
        >>> variables = ['x_CityA_1', 'x_CityB_2', 'x_CityC_3']
        >>> values = [1, 1, 1]
        >>> start = 'Munich'
        >>> extract_tour_from_solution(variables, values, start)
        ['Munich', 'CityA', 'CityB', 'CityC', 'Munich']
    """
    tour_dict: Dict[int, str] = {}

    # Start city is fixed at position 0
    tour_dict[0] = start_city

    # Extract positions for other cities from active variables
    for var, val in zip(variable_names, sample_values):
        if val == 1:
            _, city, pos = var.split('_')
            tour_dict[int(pos)] = city

    # Build tour in order of positions
    tour = [tour_dict[i] for i in sorted(tour_dict.keys())]
    tour.append(tour[0])  # Return to start to complete the cycle

    return tour


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth.

    Uses the Haversine formula to compute the shortest distance over the
    Earth's surface between two geographic coordinates, accounting for the
    Earth's curvature.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance between the two points in kilometers.

    Note:
        This function assumes a spherical Earth with radius 6371 km. For
        higher accuracy applications, consider using the Vincenty formula
        which accounts for Earth's ellipsoidal shape.

    Example:
        >>> # Distance between Munich and Berlin
        >>> haversine(48.1351, 11.5820, 52.5200, 13.4050)
        504.2
    """
    # Radius of Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differences in latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance


def calculate_distance_matrix(
        cities_dict: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Calculate distance matrix between all pairs of cities.

    Computes the pairwise distances between all cities using the Haversine
    formula and returns them in a symmetric matrix format suitable for TSP
    optimization algorithms.

    Args:
        cities_dict: Dictionary mapping city names to their coordinates as
            (latitude, longitude) tuples. Coordinates should be in decimal
            degrees format.

    Returns:
        Symmetric distance matrix as a pandas DataFrame where both row and
        column indices are city names, and values represent distances in
        kilometers (rounded to nearest integer).

    Note:
        The diagonal elements (distance from a city to itself) will be 0.
        The matrix is symmetric, so distance(A,B) equals distance(B,A).

    Example:
        >>> cities = {
        ...     'Munich': (48.1351, 11.5820),
        ...     'Berlin': (52.5200, 13.4050),
        ...     'Hamburg': (53.5511, 9.9937)
        ... }
        >>> calculate_distance_matrix(cities)
                Munich  Berlin  Hamburg
        Munich       0     504      612
        Berlin     504       0      255
        Hamburg    612     255        0
    """
    # Get the city names
    cities = list(cities_dict.keys())

    # Initialize an empty distance matrix
    distance_matrix: List[List[int]] = []

    # Compute distance for each pair of cities
    for city1 in cities:
        row = []
        for city2 in cities:
            lat1, lon1 = cities_dict[city1]
            lat2, lon2 = cities_dict[city2]
            distance = haversine(lat1, lon1, lat2, lon2)
            distance_km = int(round(distance, 0))
            row.append(distance_km)
        distance_matrix.append(row)

    # Convert the distance matrix to a pandas DataFrame for better visualization
    df = pd.DataFrame(distance_matrix, index=cities, columns=cities)
    return df