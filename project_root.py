# A simple script that allows to set the project root by importing this file
# based on https://stackoverflow.com/a/25389715

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
CACHE_DIR = "<CACHE_DIR>"

def join_with_root(path):
    return os.path.join(ROOT_DIR, path)


# flash-attn                        2.8.3