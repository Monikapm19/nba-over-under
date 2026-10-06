"""Approximate home-city coordinates (latitude, longitude, degrees) for the 30 NBA teams.

SOURCE / METHOD: general public geographic knowledge of each team's home city
(city-centre coordinates rounded to 2 decimals, i.e. ~1 km precision). These are NOT
taken from the Kaggle dataset and are NOT exact arena addresses. For a distance feature
measured in hundreds/thousands of km, a city-level error of a few km is negligible.
Teams that share a city (LAL/LAC, NY/BKN) get identical or near-identical coordinates.
Franchise moves are already normalised by the dataset (e.g. Seattle -> 'okc',
New Jersey -> 'bkn'); we use the current city for the whole period.
"""
ARENAS = {
    "atl": (33.75, -84.39), "bkn": (40.68, -73.97), "bos": (42.36, -71.06),
    "cha": (35.23, -80.84), "chi": (41.88, -87.63), "cle": (41.50, -81.69),
    "dal": (32.78, -96.80), "den": (39.74, -104.99), "det": (42.33, -83.05),
    "gs": (37.77, -122.39), "hou": (29.76, -95.37), "ind": (39.77, -86.16),
    "lac": (34.04, -118.27), "lal": (34.04, -118.27), "mem": (35.14, -90.05),
    "mia": (25.78, -80.19), "mil": (43.04, -87.92), "min": (44.98, -93.28),
    "no": (29.95, -90.08), "ny": (40.75, -73.99), "okc": (35.46, -97.52),
    "orl": (28.54, -81.38), "phi": (39.90, -75.17), "phx": (33.45, -112.07),
    "por": (45.53, -122.67), "sa": (29.43, -98.44), "sac": (38.58, -121.50),
    "tor": (43.64, -79.38), "utah": (40.77, -111.90), "wsh": (38.90, -77.02),
}
