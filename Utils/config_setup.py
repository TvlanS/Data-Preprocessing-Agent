import yaml
import pyprojroot
import os

root = pyprojroot.here()
print(root)

class Config():

    def __init__ (self):
        path_config = os.path.join(root,"config","app_config.yml")
        print(path_config)
        with open(path_config, "r", encoding = "utf-8") as f:
            config = yaml.safe_load(f)

            self.api_key = config["deepseek"]["api_key"]
            self.website_url = config["deepseek"]["website_url"]
            self.cleaning_prompt = config["prompt"]["cleaning_prompt"]
            self.missing_prompt = config["prompt"]["missing_prompt"]
            self.classification_prompt = config["prompt"]["classification_prompt"]


            