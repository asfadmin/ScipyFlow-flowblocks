"""
name: "Print packages"
requirements:
    - setuptools
inputs:
outputs:
description: "Prints all imported packages and their versions"
"""

import pkg_resources

def main():
    pkgs = pkg_resources.working_set
    list = [(pkg.project_name, pkg.version) for pkg in pkgs]
    for pkg in list:
        print(f"{pkg[0]}=={pkg[1]}")
    