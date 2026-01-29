import requests
from config.secret import GITHUB_TOKEN

r = requests.get(
    "https://api.github.com/repos/godotengine/godot/issues", 
    headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
print(r.status_code, r.json())
