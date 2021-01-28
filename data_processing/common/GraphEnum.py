from enum import Enum
from data_processing.common.utils import to_sql_field
from data_processing.common.Node import Node
from data_processing.common.Neo4jConnection import Neo4jConnection
import os, socket
	
class Container(object):
	"""
	Container: an abstraction with an id, name, namespace, type, and a list of associated data nodes

	:param: conn: Neo4j Connection
	:param: container_id: Container ID
	"""
	# TODO: worried about schema issues? like making sure name, namespace, type and qualified path are present, neo4j offers schema enforcment. 
	# TODO: testing
	# TODO: error checking

	def __init__(self, params):
		self.params=params

		print ("Connecting to:", params['GRAPH_URI'])
		self._conn = Neo4jConnection(uri=params['GRAPH_URI'], user=params['GRAPH_USER'], pwd=params['GRAPH_PASSWORD'])
		print ("Connection successfull:", self._conn.test_connection())
		self._host = socket.gethostname() # portable to *docker* containers
		print ("Running on:", self._host)
	
	def setNamespace(self, namespace_id):
		self._namespace_id = namespace_id
		return self
	
	def lookupAndAttach(self, container_id):
		self._attached = False
		if type(container_id) is str: container_id = rf"'{container_id}'"
		print ("Lookup ID:", container_id)
		self._match_clause = f"""WHERE id(container) = {container_id} OR container.QualifiedPath = {container_id}"""
		print ("Match on:", self._match_clause)
		res = self._conn.query(f"""
			MATCH (container) {self._match_clause}
			RETURN labels(container), container.type, container.name, container.Namespace, container.QualifiedPath"""
		)
		if res is None or len(res) == 0: 
			print ("Not found")
			return self

		print ("Found:", res)
		self._container_id  = container_id
		self._name 			= res[0]["container.name"]
		self._namespace     = res[0]["container.Namespace"]
		self._qualifiedpath = res[0]["container.QualifiedPath"]
		self._type		 	= res[0]["container.type"]
		self._labels		= res[0]["labels(container)"]

		if self._qualifiedpath is None: 
			print ("Found, however not valid container object, containers must have a name, namespace, and qualified path")
			return self

		print ("Successfully attached to:", self._type, self._qualifiedpath)
		self._attached = True

		return self
	
	def isAttached(self):
		print ("Attached:", self._attached)

	def get(self, type):
		res = self._conn.query(f"""
			MATCH (container)-[:HAS_DATA]-(data:{type}) 
			{self._match_clause}
			RETURN data"""
		)
		if res is None or len(res) == 0: 
			return None
		else:
		    return [Node(rec['data']['type'], rec['data']['name'], dict(rec['data'].items())) for rec in res]


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
	accession_radiology_dicom = NodeType("accession", "metadata.AccessionNumber")
	accession_radiology = NodeType("accession", "accession_number")
	scan = NodeType("scan", "metadata.SeriesInstanceUID")
	png = NodeType("png", "png_record_uuid", ["metadata.SeriesInstanceUID", "label"])

	DICOM = [Graph(NodeType("patient", "metadata.PatientID"),
				"HAS_CASE", 
				accession_radiology_dicom),
			Graph(accession_radiology_dicom,
				"HAS_SCAN", 
				scan)]

	MHA = [Graph(accession_radiology,
				"HAS_DATA",
				NodeType("mha", "scan_annotation_record_uuid", ["path", "label"]))]

	MHD = [Graph(accession_radiology,
				"HAS_DATA",
				NodeType("mhd", "scan_annotation_record_uuid", ["path", "label"]))]

	PNG = [Graph(scan, "HAS_DATA", png)]

	FEATURE = [Graph(png, "HAS_DATA",
			NodeType("feature", "feature_record_uuid", ["metadata.SeriesInstanceUID", "label"]))]

