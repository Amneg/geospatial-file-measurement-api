import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import BadZipFile, LargeZipFile, ZipFile

import geopandas as gpd
import pyogrio


MAX_EXTRACTED_SIZE = 100 * 1024 * 1024


def extract_shapefile(zip_path: Path, temporary_directory: Path):
    try:
        with ZipFile(zip_path) as archive:
            members = archive.infolist()
            if sum(member.file_size for member in members) > MAX_EXTRACTED_SIZE:
                raise ValueError("ZIP contents exceed the 100 MB extraction limit.")

            files = {}
            for member in members:
                path = PurePosixPath(member.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.filename
                    or ":" in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16)
                ):
                    raise ValueError("ZIP contains an unsafe path or symbolic link.")
                if member.is_dir():
                    continue
                name = path.as_posix().lower()
                if name in files:
                    raise ValueError("ZIP contains duplicate file paths.")
                files[name] = member

            shapefiles = [name for name in files if name.endswith(".shp")]
            if len(shapefiles) != 1:
                raise ValueError("ZIP must contain exactly one .shp file.")
            shapefile = PurePosixPath(shapefiles[0])
            for extension in (".shx", ".dbf", ".prj"):
                if shapefile.with_suffix(extension).as_posix() not in files:
                    raise ValueError(f"Shapefile is missing a matching {extension} file.")
            if archive.testzip() is not None:
                raise ValueError("ZIP contains corrupt file data.")

            # Copy only validated components to fixed names, never extract arbitrary paths.
            for extension in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                name = shapefile.with_suffix(extension).as_posix()
                if name in files:
                    destination = temporary_directory / f"dataset{extension}"
                    with archive.open(files[name]) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
            return temporary_directory / "dataset.shp"
    except (BadZipFile, LargeZipFile, RuntimeError, NotImplementedError) as error:
        raise ValueError("Corrupt or unsupported ZIP archive.") from error


def read_features(dataset_path: Path):
    try:
        layers = pyogrio.list_layers(dataset_path)
        if len(layers) == 0:
            raise ValueError("The dataset does not contain any readable layers.")
        driver = pyogrio.read_info(dataset_path, layer=layers[0][0])["driver"]
        if dataset_path.suffix.lower() == ".kml":
            allowed_drivers = {"KML", "LIBKML"}
        else:
            allowed_drivers = {"ESRI Shapefile"}
        if driver not in allowed_drivers:
            raise ValueError("The file contents do not match the expected geospatial format.")
        frames = [
            gpd.read_file(dataset_path, layer=layer_name, engine="pyogrio", fid_as_index=True)
            for layer_name, _ in layers
        ]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Could not read the geospatial file. Check that the dataset is valid.") from error

    source_crs = frames[0].crs
    if source_crs is None:
        raise ValueError("The dataset CRS is missing or invalid.")

    features = []
    for frame in frames:
        if frame.crs != source_crs:
            raise ValueError("All layers must have the same source CRS.")
        # Keep source coordinates; turn missing attributes into null and dates into text.
        data = json.loads(frame.to_json(na="null", to_wgs84=False, default=str))
        for feature in data["features"]:
            geometry = feature["geometry"]
            features.append({
                "source_feature_id": feature["id"],
                "geometry_type": geometry["type"] if geometry is not None else "Unknown",
                "geometry": geometry,
                "properties": feature["properties"],
            })
    return source_crs.to_string(), features
