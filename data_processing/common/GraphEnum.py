from enum import Enum


def does_not_contain(token, value):
	"""
	Validate that `token` is not a substring of `value`

	:param: token: string e.g. : | .
	:param: value: dictionary, list, or str
	"""
	if isinstance(value, str):
		if token in value:
			raise RuntimeError("{value} cannot contain {token}.")

    if isinstance(value, list):
    	if any([token in v for v in value]):
    		raise RuntimeError(str(value) + " cannot contain {token}.")

	if isinstance(value, dict):
		if [token in k or token in v for k,v in value.items()]
			raise RuntimeError(str(value) + " cannot contain {token}.")

	return True


def replace_token(token, token_replacement, value):
	"""
	Replace `token` with `token_replacement` in `value`

	:param: token: string e.g. : | .
	:param: token_replacement: string e.g. _ -
	:param: value: dictionary, list, or str
	"""
	if isinstance(value, str):
		return value.replace(token, token_replacement)

    if isinstance(value, list):
    	new_value = []
    	for v in value:
    		new_value.append(v.replace(token, token_replacement))
    	return new_value

	if isinstance(value, dict):
		new_value = {}
		for k,v in value.items():
			new_value[k.replace(token, token_replacement)] = v.replace(token, token_replacement)

		return new_value

	return value
	

class NodeType(object):
	"""
	NodeType object defines the schema used to populate a graph node.

	:param: node_type: node type. e.g. scan
	:param: schema: list of column names. e.g. slide_id
	"""
	def __init__(self, node_type, schema=[], properties={}):

		self.type = node_type
		self.fields = fields

	def create(self):
		prop_string = self.prop_str(self.properties.keys(), self.properties)
		return f"""{self.type}:globals{{ {prop_string} }}"""

	def match(self):
		prop_string = self.prop_str( ["QualifiedPath"], self.properties)
		return f"""{self.type}{{ {prop_string} }}"""

	@staticmethod
	def prop_str(fields, row):
		"""
		Returns a kv string like 'id: 123, ...' where prop values come from row.
		"""
		fields = set(fields).intersection(set(row.keys()))

		kv = [f" {x}: '{row[x]}'" for x in fields]
		return ','.join(kv)


class Graph(object):
	"""
	Graph object that stores src-[relationship]-target information.
	This object is used get data from existing delta tables to populate the graph database.

	:param: src: NodeType - source node
	:param: relationship: str - relationship name e.g. HAS_PX, HAS_RECORD etc
	:param: target: NodeType - target node
	"""
	def __init__(self, src, relationship, target):

		self.src = src
		self.relationship = relationship
		self.target = target


class GraphEnum(Enum):
	"""
	Defines Graph relationships. 
	We assume that all properties come from the table defined in GraphEnum.name

	name: table name
	value: list of Graphs - to accomodate multiple relationship update

	>>> GraphEnum['DICOM'].value[0].src.type
	'xnat_patient'
	"""
	DICOM = [Graph(NodeType("xnat_patient", ["metadata.PatientID"]),
			"HAS_SCAN", 
			NodeType("scan", ["metadata.SeriesInstanceUID"]))]
	#PROJECT = Graph("project_name", "HAS_PX", "dmp_patient_id")
