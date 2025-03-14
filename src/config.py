import yaml

with open('config.yml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)