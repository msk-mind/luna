from enum import Enum

class Node(object):
	"""
	Node object defines the type and attributes of a graph node.

	:param: node_type: node type. e.g. scan
	:param: fields: list of column names. e.g. slide_id
	"""
	def __init__(self, node_type, fields):

		self.type = node_type
		self.fields = fields

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
