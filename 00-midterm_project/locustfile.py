from locust import task
from locust import between
from locust import HttpUser

sample = {
  "relative_compactness": 0.790000,
  "surface_area": 637.000000,
  "wall_area": 343.000000,
  "roof_area": 147.000000,
  "overall_height": 7.0,
  "orientation": 5,
  "glazing_area": 0.0,
  "glazing_area_distribution": 0
}

class MLZoomUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:

            locust -H http://localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)
