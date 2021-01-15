from enum import Enum
from data_processing.common.utils import to_sql_field
	

class NodeType(object):
	"""
	NodeType object defines the schema used to populate a graph node.

	:param: node_type: node type. e.g. scan
	:param: name: required node name. e.g. scan-123
	:param: schema: list of column names. e.g. slide_id
	"""
	def __init__(self, node_type, name, schema=[]):

		self.type = node_type
		self.name = name
		self.schema = schema


	def get_all_schema(self):
		"""
		Name is a required field, but it's still a property of this node.
		Return the properties including the name property!
		"""
		return self.schema + [self.name]


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
	'patient'
	"""
	DICOM = [Graph(NodeType("patient", "metadata.PatientID"),
				"HAS_CASE", 
				NodeType("accession", "metadata.AccessionNumber")),
			Graph(NodeType("accession", "metadata.AccessionNumber"),
				"HAS_SCAN", 
				NodeType("scan", "metadata.SeriesInstanceUID"))]

	MHA = [Graph(NodeType("accession", "accession_number"),
				"HAS_DATA",
				NodeType("mha", "scan_annotation_record_uuid", ["path", "label"]))]

	MHD = [Graph(NodeType("accession", "accession_number"),
				"HAS_DATA",
				NodeType("mhd", "scan_annotation_record_uuid", ["path", "label"]))]

	PNG = [Graph(NodeType("accession", "accession_number"),
				"HAS_DATA",
				NodeType("png", "png_record_uuid", ["metadata.SeriesInstanceUID", "label"]))]

	FEATURE = [Graph(NodeType("accession", "accession_number"),
			"HAS_DATA",
			NodeType("feature", "feature_record_uuid", ["metadata.SeriesInstanceUID", "label"]))]

