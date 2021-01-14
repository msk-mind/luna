from enum import Enum
from data_processing.common.utils import to_sql_field
	

class NodeType(object):
	"""
	NodeType object defines the schema used to populate a graph node.

	:param: node_type: node type. e.g. scan
	:param: schema: list of column names. e.g. slide_id
	"""
	def __init__(self, node_type, schema=[]):

		self.type = node_type
		self.schema = schema


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
