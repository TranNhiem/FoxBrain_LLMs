"""
Author: Po-Kai Chen
Date: 1/8/2024
"""
import json
import os


root_path = os.path.dirname(os.path.abspath(__file__))


def get_prompt(task):
    with open(f"{root_path}/{task}_prompts.json", "r") as r:
        return json.load(r)