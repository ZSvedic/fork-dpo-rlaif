import yaml
import json

with open("config.yaml", "r") as yaml_file:
    config_data = yaml.safe_load(yaml_file)

with open("config.json", "w") as json_file:
    json.dump(config_data, json_file, indent=4)
