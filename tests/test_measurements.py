import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

from app.geo import calculate_measurements


def test_geographic_polygon_area_in_square_meters():
    polygon = Polygon([(3, 0), (3.001, 0), (3.001, 0.001), (3, 0.001)])
    geometries = gpd.GeoSeries([polygon], crs="EPSG:4326")
    result = calculate_measurements(geometries)[0]
    # At the equator, these sides are about 111.32 m and 110.57 m.
    assert result["area_m2"] == pytest.approx(12309, rel=0.003)
    assert result["length_m"] is None
    assert result["measurement_status"] == "SUCCESS"
    assert geometries.iloc[0].equals_exact(polygon, tolerance=0)
    assert geometries.crs.to_epsg() == 4326


def test_geographic_line_length_in_meters():
    line = LineString([(3, 0), (3.001, 0)])
    result = calculate_measurements(gpd.GeoSeries([line], crs="EPSG:4326"))[0]
    assert result["length_m"] == pytest.approx(111.32, rel=0.002)
    assert result["area_m2"] is None
    assert result["measurement_status"] == "SUCCESS"


def test_point_needs_no_measurement():
    result = calculate_measurements(gpd.GeoSeries([Point(3, 0)], crs="EPSG:4326"))[0]
    assert result == {"area_m2": None, "length_m": None, "measurement_status": "NOT_REQUIRED"}


@pytest.mark.parametrize("unsupported", [
    MultiPolygon([Polygon([(3, 0), (3.001, 0), (3.001, 0.001)])]),
    MultiLineString([[(3, 0), (3.001, 0)]]),
    MultiPoint([(3, 0), (3.001, 0)]),
    GeometryCollection([Point(3, 0)]),
])
def test_unsupported_geometry_does_not_interrupt_other_features(unsupported):
    line = LineString([(3, 0), (3.001, 0)])
    results = calculate_measurements(gpd.GeoSeries([unsupported, line], crs="EPSG:4326"))
    assert results[0] == {"area_m2": None, "length_m": None, "measurement_status": "UNSUPPORTED"}
    assert results[1]["length_m"] == pytest.approx(111.32, rel=0.002)
    assert results[1]["measurement_status"] == "SUCCESS"


def test_missing_crs_is_a_clear_error():
    geometries = gpd.GeoSeries([LineString([(3, 0), (3.001, 0)])])
    with pytest.raises(ValueError, match="CRS is missing or invalid"):
        calculate_measurements(geometries)


def test_unusable_crs_is_a_clear_error():
    geometries = gpd.GeoSeries([Point(1, 2)], crs="EPSG:4978")
    with pytest.raises(ValueError, match="CRS is not usable"):
        calculate_measurements(geometries)


def test_projected_meter_crs_area_and_length():
    polygon = Polygon([(500000, 0), (500100, 0), (500100, 100), (500000, 100)])
    line = LineString([(500000, 0), (500100, 0)])
    results = calculate_measurements(gpd.GeoSeries([polygon, line], crs="EPSG:32631"))
    assert results[0]["area_m2"] == pytest.approx(10000)
    assert results[0]["length_m"] is None
    assert results[1]["length_m"] == pytest.approx(100)
    assert results[1]["area_m2"] is None
    assert all(result["measurement_status"] == "SUCCESS" for result in results)


def test_projected_feet_crs_is_measured_in_meters():
    polygon = Polygon([(987000, 210000), (987100, 210000), (987100, 210100), (987000, 210100)])
    line = LineString([(987000, 210000), (987100, 210000)])
    results = calculate_measurements(gpd.GeoSeries([polygon, line], crs="EPSG:2263"))
    assert results[0]["area_m2"] == pytest.approx(929.03, rel=0.003)
    assert results[1]["length_m"] == pytest.approx(30.48, rel=0.002)
    assert all(result["measurement_status"] == "SUCCESS" for result in results)


def test_invalid_polygon_does_not_interrupt_valid_line():
    polygon = Polygon([(500000, 0), (500100, 100), (500100, 0), (500000, 100)])
    line = LineString([(500000, 0), (500100, 0)])
    results = calculate_measurements(gpd.GeoSeries([polygon, line], crs="EPSG:32631"))
    assert results[0] == {"area_m2": None, "length_m": None, "measurement_status": "INVALID_GEOMETRY"}
    assert results[1]["length_m"] == pytest.approx(100)
