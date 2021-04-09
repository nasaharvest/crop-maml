from dataclasses import dataclass
from math import sin, cos, sqrt, atan2, radians


@dataclass
class BoundingBox:

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    def distance_from(self, box) -> float:
        r"""
        Another bounding box as input. Return
        approximate distance in km
        """
        # approximate radius of earth in km
        earth_radius = 6373.0

        # roughly calculate the centres
        lat1 = radians(self.min_lat + ((self.max_lat - self.min_lat) / 2))
        lon1 = radians(self.min_lon + ((self.max_lon - self.min_lon) / 2))

        lat2 = radians(box.min_lat + ((box.max_lat - box.min_lat) / 2))
        lon2 = radians(box.min_lon + ((box.max_lon - box.min_lon) / 2))

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return earth_radius * c


STR2BB = {
    "Kenya": BoundingBox(min_lon=33.501, max_lon=42.283, min_lat=-5.202, max_lat=6.002),
    # these are rough; eyeballed from google earth
    "Busia": BoundingBox(
        min_lon=33.88389587402344,
        min_lat=-0.04119872691853491,
        max_lon=34.44007873535156,
        max_lat=0.7779454563313616,
    ),
    "Togo": BoundingBox(
        min_lon=-0.1501, max_lon=1.7779296875, min_lat=6.08940429687, max_lat=11.115625,
    ),
    "West Bahia": BoundingBox(min_lon=-47, max_lon=-44.5, min_lat=-15, max_lat=-5),
}
