import os


def pytest_configure():
    # Ensure config parsing doesn't fail during test collection if the developer
    # environment has an invalid PORT value set.
    os.environ["PORT"] = "8081"
