"""Runtime utilities for the archived HFH protocol."""
import json

import zipfile

def zip_read(path, name):
    with zipfile.ZipFile(path) as stream:
        return json.loads(stream.read('results/'+name))
