import requests
import pandas as pd
from config.secret import GITHUB_TOKEN

r = requests.get(
    "https://api.github.com/repos/godotengine/godot/issues", 
    headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
df = pd.DataFrame(r.json())
file_path = "out.json"
df.to_json(file_path, orient="records", indent=4)
