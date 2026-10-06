from io import BytesIO
import stat
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from fastapi.testclient import TestClient
import geopandas as gpd
import pytest
from shapely.geometry import Point
from sqlalchemy import create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app import geo, main, models
from app.database import Base, get_db


KML_DATA = b'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><name>Station</name><Point><coordinates>77,28,0</coordinates></Point></Placemark>
  <Placemark><name>Road</name><LineString><coordinates>77,28,0 77.1,28.1,0</coordinates></LineString></Placemark>
  <Placemark><name>Park</name><Polygon><outerBoundaryIs><LinearRing>
    <coordinates>77,28,0 77.1,28,0 77.1,28.1,0 77,28,0</coordinates>
  </LinearRing></outerBoundaryIs></Polygon></Placemark>
</Document></kml>'''


@pytest.fixture
def api(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False},
    )
    sessions = sessionmaker(bind=test_engine)
    upload_directory = tmp_path / "uploads"
    monkeypatch.setattr(main, "engine", test_engine)
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_directory)

    def test_db():
        with sessions() as db:
            yield db

    main.app.dependency_overrides[get_db] = test_db
    try:
        with TestClient(main.app) as client:
            yield client, sessions, upload_directory
    finally:
        main.app.dependency_overrides.clear()
        test_engine.dispose()


@pytest.fixture
def shapefile_parts(tmp_path):
    frame = gpd.GeoDataFrame(
        {"label": ["Station", "Office"], "count": [3, 7], "optional": [None, "yes"]},
        geometry=[Point(77, 28), Point(78, 29)], crs="EPSG:4326",
    )
    directory = tmp_path / "shapefile"
    directory.mkdir()
    frame.to_file(directory / "places.shp", driver="ESRI Shapefile", engine="pyogrio")
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def make_zip(parts):
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_STORED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return output.getvalue()


def upload(client, filename, data):
    return client.post("/api/files/", files={"file": (filename, data)})


def assert_no_saved_data(sessions, upload_directory):
    with sessions() as db:
        assert db.query(models.File).count() == 0
        assert db.query(models.Feature).count() == 0
    assert list(upload_directory.glob("*")) == []


def test_health(api):
    client, _, _ = api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_kml_upload_and_feature_extraction(api):
    client, sessions, upload_directory = api
    response = upload(client, "sample.kml", KML_DATA)
    assert response.status_code == 201, response.text
    metadata = response.json()
    assert metadata["id"] > 0
    assert metadata["filename"] == "sample.kml"
    assert metadata["source_crs"] == "EPSG:4326"
    assert metadata["feature_count"] == 3
    assert metadata["status"] == "COMPLETED"
    assert metadata["uploaded_at"]
    with sessions() as db:
        stored_file = db.get(models.File, metadata["id"])
        assert stored_file.source_crs == "EPSG:4326"
        assert stored_file.status == "COMPLETED"
        features = db.query(models.Feature).order_by(models.Feature.id).all()
        assert [feature.geometry_type for feature in features] == ["Point", "LineString", "Polygon"]
        assert len({feature.source_feature_id for feature in features}) == 3
        assert features[0].geometry["coordinates"][:2] == [77, 28]
        assert features[0].properties["Name"] == "Station"
        assert features[1].geometry == {
            "type": "LineString", "coordinates": [[77, 28, 0], [77.1, 28.1, 0]],
        }
        assert features[2].geometry == {
            "type": "Polygon", "coordinates": [[[77, 28, 0], [77.1, 28, 0], [77.1, 28.1, 0], [77, 28, 0]]],
        }
        for feature in features:
            assert feature.file_id == metadata["id"]
        assert features[0].area_m2 is None and features[0].length_m is None
        assert features[0].measurement_status == "NOT_REQUIRED"
        assert features[1].length_m > 0 and features[1].area_m2 is None
        assert features[1].measurement_status == "SUCCESS"
        assert features[2].area_m2 > 0 and features[2].length_m is None
        assert features[2].measurement_status == "SUCCESS"
    saved_files = list(upload_directory.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].name == f"{metadata['id']}.kml"
    assert saved_files[0].read_bytes() == KML_DATA


@pytest.mark.parametrize("uppercase", [False, True])
def test_valid_nested_shapefile_zip_and_attributes(api, shapefile_parts, uppercase):
    client, sessions, upload_directory = api
    parts = {f"nested/{name.upper() if uppercase else name}": data for name, data in shapefile_parts.items()}
    data = make_zip(parts)
    response = upload(client, "places.ZIP", data)
    assert response.status_code == 201, response.text
    assert response.json()["feature_count"] == 2
    assert response.json()["source_crs"] == "EPSG:4326"
    with sessions() as db:
        features = db.query(models.Feature).order_by(models.Feature.id).all()
        assert [feature.source_feature_id for feature in features] == ["0", "1"]
        assert features[0].geometry == {"type": "Point", "coordinates": [77.0, 28.0]}
        assert features[0].properties == {"label": "Station", "count": 3, "optional": None}
    assert next(upload_directory.iterdir()).read_bytes() == data


def test_unique_ids_and_saved_names(api):
    client, sessions, upload_directory = api
    first = upload(client, "same.kml", KML_DATA)
    second = upload(client, "same.kml", KML_DATA)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    assert len(list(upload_directory.iterdir())) == 2
    with sessions() as db:
        assert db.query(models.File).count() == 2
        assert db.query(models.Feature).count() == 6


@pytest.mark.parametrize("filename", ["sample.txt", "sample.geojson", "sample.shp"])
def test_unsupported_extension(api, filename):
    client, sessions, upload_directory = api
    response = upload(client, filename, b"some data")
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


@pytest.mark.parametrize("filename", ["empty.kml", "empty.zip"])
def test_empty_upload(api, filename):
    client, sessions, upload_directory = api
    response = upload(client, filename, b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_corrupt_zip(api):
    client, sessions, upload_directory = api
    response = upload(client, "corrupt.zip", b"not a ZIP archive")
    assert response.status_code == 400
    assert "Corrupt" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_corrupt_zip_member(api):
    client, sessions, upload_directory = api
    parts = {f"roads{extension}": b"dummy" for extension in (".shp", ".shx", ".dbf", ".prj")}
    data = make_zip(parts).replace(b"dummy", b"xxxxx", 1)
    response = upload(client, "corrupt.zip", data)
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


@pytest.mark.parametrize("extension", [".shx", ".dbf", ".prj"])
def test_missing_shapefile_component(api, shapefile_parts, extension):
    client, sessions, upload_directory = api
    del shapefile_parts[f"places{extension}"]
    response = upload(client, "missing.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert f"matching {extension}" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_components_must_match_shapefile_name(api, shapefile_parts):
    client, sessions, upload_directory = api
    shapefile_parts["other.prj"] = shapefile_parts.pop("places.prj")
    response = upload(client, "mismatched.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert "matching .prj" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


@pytest.mark.parametrize("count", [0, 2])
def test_exactly_one_shapefile_required(api, shapefile_parts, count):
    client, sessions, upload_directory = api
    if count == 0:
        shapefile_parts.pop("places.shp")
    else:
        shapefile_parts["other.shp"] = shapefile_parts["places.shp"]
    response = upload(client, "invalid.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert "exactly one .shp" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


@pytest.mark.parametrize("unsafe_name", ["../escape.txt", "/escape.txt", "C:/escape.txt", "nested/../../escape.txt", "nested" + chr(92) + "escape.txt"])
def test_unsafe_zip_paths(api, shapefile_parts, unsafe_name):
    client, sessions, upload_directory = api
    shapefile_parts[unsafe_name] = b"unsafe"
    response = upload(client, "unsafe.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert "unsafe" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_zip_symbolic_link(api, shapefile_parts):
    client, sessions, upload_directory = api
    link = ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    shapefile_parts[link] = b"../outside"
    response = upload(client, "symlink.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert "symbolic link" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_invalid_kml_leaves_no_records(api):
    client, sessions, upload_directory = api
    response = upload(client, "invalid.kml", b"not XML")
    assert response.status_code == 400
    assert "Could not read" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_projected_source_crs_and_coordinates_are_preserved(api, tmp_path):
    client, sessions, _ = api
    frame = gpd.GeoDataFrame({"name": ["Survey point"]}, geometry=[Point(500000, 3000000)], crs="EPSG:32643")
    directory = tmp_path / "projected"
    directory.mkdir()
    frame.to_file(directory / "survey.shp", driver="ESRI Shapefile", engine="pyogrio")
    parts = {path.name: path.read_bytes() for path in directory.iterdir()}
    response = upload(client, "survey.zip", make_zip(parts))
    assert response.status_code == 201, response.text
    assert response.json()["source_crs"] == "EPSG:32643"
    with sessions() as db:
        feature = db.query(models.Feature).one()
        assert feature.geometry["coordinates"] == [500000, 3000000]
        assert feature.area_m2 is None and feature.length_m is None
        assert feature.measurement_status == "NOT_REQUIRED"


def test_all_kml_layers_are_read(api):
    client, sessions, _ = api
    data = b'''<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
      <Folder><name>First</name><Placemark><Point><coordinates>77,28</coordinates></Point></Placemark></Folder>
      <Folder><name>Second</name><Placemark><Point><coordinates>78,29</coordinates></Point></Placemark></Folder>
    </Document></kml>'''
    response = upload(client, "folders.kml", data)
    assert response.status_code == 201, response.text
    assert response.json()["feature_count"] == 2
    with sessions() as db:
        assert db.query(models.Feature).count() == 2


def test_upload_size_limit(api, monkeypatch):
    client, sessions, upload_directory = api
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE", 10)
    response = upload(client, "large.kml", KML_DATA)
    assert response.status_code == 413
    assert_no_saved_data(sessions, upload_directory)


def test_zip_extraction_size_limit(api, shapefile_parts, monkeypatch):
    client, sessions, upload_directory = api
    monkeypatch.setattr(geo, "MAX_EXTRACTED_SIZE", 10)
    response = upload(client, "large.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert "extraction limit" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_database_failure_rolls_back_file_and_features(api):
    client, sessions, upload_directory = api

    def fail_feature_insert(session, flush_context, instances):
        if any(isinstance(row, models.Feature) for row in session.new):
            raise SQLAlchemyError("Simulated feature insert failure")

    event.listen(sessions, "before_flush", fail_feature_insert)
    try:
        response = upload(client, "sample.kml", KML_DATA)
    finally:
        event.remove(sessions, "before_flush", fail_feature_insert)
    assert response.status_code == 500
    assert "Could not save" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


@pytest.mark.parametrize("placemark_geometry, expected_type", [
    ("", "Unknown"),
    ("<MultiGeometry><Point><coordinates>77,28</coordinates></Point><LineString><coordinates>77,28 78,29</coordinates></LineString></MultiGeometry>", "GeometryCollection"),
])
def test_missing_and_collection_geometries_are_preserved(api, placemark_geometry, expected_type):
    client, sessions, _ = api
    data = f'<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Example</name>{placemark_geometry}</Placemark></Document></kml>'.encode()
    response = upload(client, "geometry.kml", data)
    assert response.status_code == 201, response.text
    with sessions() as db:
        feature = db.query(models.Feature).one()
        assert feature.geometry_type == expected_type
        if expected_type == "Unknown":
            assert feature.geometry is None
        else:
            assert feature.geometry["type"] == "GeometryCollection"
        assert feature.area_m2 is None and feature.length_m is None
        assert feature.measurement_status == "UNSUPPORTED"


def test_invalid_shapefile_crs(api, shapefile_parts):
    client, sessions, upload_directory = api
    shapefile_parts["places.prj"] = b"not a valid CRS"
    response = upload(client, "invalid-crs.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert "CRS" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_zip_duplicate_paths(api, shapefile_parts):
    client, sessions, upload_directory = api
    shapefile_parts["PLACES.SHP"] = shapefile_parts["places.shp"]
    response = upload(client, "duplicate.zip", make_zip(shapefile_parts))
    assert response.status_code == 400
    assert "duplicate" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_disguised_non_kml_is_rejected(api):
    client, sessions, upload_directory = api
    data = b'{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[77,28]},"properties":{}}]}'
    response = upload(client, "disguised.kml", data)
    assert response.status_code == 400
    assert response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)


def test_measurement_crs_failure_cleans_up_upload(api):
    client, sessions, upload_directory = api
    data = b'''<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
      <Placemark><LineString><coordinates>3,85 3.001,85</coordinates></LineString></Placemark>
    </Document></kml>'''
    response = upload(client, "polar.kml", data)
    assert response.status_code == 400
    assert "UTM CRS" in response.json()["detail"]
    assert_no_saved_data(sessions, upload_directory)
