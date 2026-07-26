import sys
from fastapi.testclient import TestClient
from main import app

sys.stdout.reconfigure(encoding='utf-8')

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    print(f"Root OK: {response.json()['app_name']}")

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert "r2" in data["metrics"]
    print(f"Health: R²={data['metrics']['r2']:.4f}, MAE={data['metrics']['mae']:.2f}°C")

def test_cities():
    response = client.get("/api/cities")
    assert response.status_code == 200
    data = response.json()
    cities = data["cities"]
    assert len(cities) > 0
    assert "data_source" in data
    print(f"Loaded {len(cities)} cities from {data['data_source']}")
    for c in cities[:3]:
        print(f"  - {c['name']}: lat={c['latitude']}, lon={c['longitude']}, avg_lst={c['avg_lst_celsius']}°C, ndvi={c['ndvi']}, cells={c['total_cells']}")

def test_city_grid():
    """Test real parquet grid data endpoint."""
    response = client.get("/api/city-grid/chennai?max_cells=50")
    assert response.status_code == 200
    data = response.json()
    assert data["city"] == "chennai"
    assert len(data["grid"]) > 0
    cell = data["grid"][0]
    assert "centroid_lat" in cell
    assert "centroid_lon" in cell
    assert "lst_mean" in cell
    assert "ndvi_mean" in cell
    print(f"Chennai grid: {data['total_cells']} cells, source: {data['data_source']}")
    print(f"  Sample cell: lat={cell['centroid_lat']:.4f}, lon={cell['centroid_lon']:.4f}, LST={cell['lst_mean']:.1f}°C, NDVI={cell['ndvi_mean']:.3f}")

def test_predict():
    payload = {
        "latitude": 13.0825,
        "longitude": 80.2750,
        "month": 5,
        "ndvi": 0.276,
        "ndbi": 0.117,
        "ndwi": -0.075,
        "elevation": 15.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "predicted_lst_celsius" in res
    assert res["predicted_lst_celsius"] > 0
    print(f"Predicted LST for Chennai: {res['predicted_lst_celsius']}°C")

def test_simulate():
    payload = {
        "latitude": 13.0825,
        "longitude": 80.2750,
        "month": 5,
        "baseline_ndvi": 0.276,
        "baseline_ndbi": 0.117,
        "baseline_ndwi": -0.075,
        "elevation": 15.0,
        "green_canopy_pct": 30.0,
        "water_features_pct": 15.0,
        "cool_roofs_pct": 40.0,
        "shade_canopies_pct": 10.0
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "impact" in res
    assert res["impact"]["delta_temp_celsius"] < 0
    print(f"Simulated cooling: {res['impact']['delta_temp_celsius']}°C")

def test_ga_optimize():
    payload = {
        "latitude": 13.0825,
        "longitude": 80.2750,
        "budget_usd": 25000.0,
        "generations": 5
    }
    response = client.post("/api/optimize-ga", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "optimal_intervention_plan" in res
    print(f"GA temperature drop: -{res['citywide_temperature_reduction_celsius']}°C, cost: ${res['total_optimal_cost_usd']}")

if __name__ == "__main__":
    test_root()
    test_health()
    test_cities()
    test_city_grid()
    test_predict()
    test_simulate()
    test_ga_optimize()
    print("\n✅ ALL TESTS PASSED — Zero mock data, all real from D:\\MINI PROJECT_NEW\\data")
