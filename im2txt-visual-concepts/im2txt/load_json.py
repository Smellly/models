import json
from ast import literal_eval

def getValID(path):
  with open(path, 'r') as f:
    raw = json.load(f)
  return raw['images']

def getValAttr(path):
  with open(path, 'r') as f:
    attr_data = json.load(f)
  filename_to_attribute = {}
  for filename, attribute in attr_data.iteritems():
    p = [literal_eval(i.split(':')[1])[0] for i in attribute]
    filename_to_attribute[filename] = p 
  return filename_to_attribute

def getNametoId(id_to_name_path):
    with open(id_to_name_path, 'r') as f:
        raw = json.load(f)
    id_to_filename = {}
    for i,j in raw.iteritems():
        id_to_filename[j] = i
    return id_to_filename

