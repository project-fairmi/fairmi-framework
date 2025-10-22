import yaml

with open('src/projects/group_similarity/config.yml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)