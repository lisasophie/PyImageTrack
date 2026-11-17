import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


@pytest.mark.parametrize("minimum_ratio", [0.0, 0.1])
def test_tracking_quality_counts_only_matches_passing_threshold(monkeypatch, minimum_ratio):
    module = importlib.import_module("PyImageTrack.ImageTracking.ImagePair")
    pair = module.ImagePair.__new__(module.ImagePair)
    area = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:32632")
    pair.crs = area.crs
    pair.undistort_image = False
    pair.images_aligned = True
    pair.safe_image_bounds_tracking = area
    pair._effective_pixel_size = Mock(return_value=1.0)
    pair.tracking_parameters = SimpleNamespace(
        distance_of_tracked_points_px=1,
        minimum_valid_tracking_point_ratio=minimum_ratio,
    )
    pair.filter_parameters = SimpleNamespace(cross_correlation_threshold_movement=0.8)
    pair.image1_matrix = pair.image2_matrix = None
    pair.image1_transform = None
    pair.image1_valid_mask = pair.image2_valid_mask = None
    pair.convert_to_3d_displacement = False
    pair.time_between_observations = 1
    pair.output_units_mode = "total"
    # All 100 rows have a correlation value, but only 14 pass the threshold.
    tracked = pd.DataFrame({"correlation_coefficient": [0.9] * 14 + [0.2] * 86})
    tracked.attrs.update(original_point_count=100, accepted_point_count=14)
    monkeypatch.setattr(module, "grid_points_on_polygon_by_distance", Mock(return_value=range(100)))
    monkeypatch.setattr(module, "track_movement_lsm", Mock(return_value=tracked))
    georeference = Mock(return_value=tracked)
    monkeypatch.setattr(module, "georeference_tracked_points", georeference)

    pair.track_points(area)

    assert pair.tracking_quality["original_point_count"] == 100
    assert pair.tracking_quality["accepted_point_count"] == 14
    assert pair.tracking_quality["accepted_ratio"] == pytest.approx(0.14)
    assert len(georeference.call_args.kwargs["tracked_pixels"]) == 100
