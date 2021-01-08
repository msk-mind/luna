from enum import Enum


class Node(object):
	"""
	Node object defines the type and attributes of a graph node.

	:param: node_type: node type. e.g. scan
	:param: fields: list of column names. e.g. slide_id
	"""
	def __init__(self, node_type, fields=[], properties={}):

		self.type = node_type
		self.fields = fields
		self.properties = properties

		if self.type=="cohort":
			if not "CohortID" in properties.keys():
				raise RuntimeError("Cohorts must have a CohortID!")
			self.properties["QualifiedPath"] = self.get_qualified_name(properties["CohortID"], properties["CohortID"])

		if self.type=="patient":
			if not ("PatientID" in properties.keys() and "Namespace" in properties.keys()):
				raise RuntimeError("Patients must have a PatientID and Namespace property!")
			self.properties["QualifiedPath"] = self.get_qualified_name(properties["Namespace"], properties["PatientID"])

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
	
	@staticmethod
	def get_qualified_name(namespace, identifier): 
		"""
		Returns the full name given a namespace and patient ID
		"""
		if ":" in namespace or ":" in identifier: raise ValueError("Qualified path cannot be constructed, namespace or identifier cannot contain ':'")
		return f"{namespace}::{identifier}"
	


class Graph(object):
	"""
	Graph object that stores src-[relationship]-target information.
	This object is used get data from existing delta tables to populate the graph database.

	:param: src: Node - source node
	:param: relationship: str - relationship name e.g. HAS_PX, HAS_RECORD etc
	:param: target: Node - target node
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
	DICOM = [Graph(Node("xnat_patient", ["metadata.PatientName"]),
			"HAS_SCAN", 
			Node("scan", ["metadata.SeriesInstanceUID"]))]
	#PROJECT = Graph("project_name", "HAS_PX", "dmp_patient_id")
