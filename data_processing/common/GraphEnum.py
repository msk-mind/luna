from enum import Enum
from data_processing.common.utils import to_sql_field
from data_processing.common.Node import Node
from data_processing.common.Neo4jConnection import Neo4jConnection
import os, socket
	
class Container(object):
	"""
	Container: an abstraction with an id, name, namespace, type, and a list of associated data nodes

	Interfaces with a metadata store (graph DB) and raw file stores (gpfs, potentially others)

	Handles the matching and creation of metadata
	"""
	# TODO: worried about schema issues? like making sure name, namespace, type and qualified path are present, neo4j offers schema enforcment. 
	# TODO: testing
	# TODO: error checking

	def __init__(self, params):
		"""
		Initialize the container object.
			Connects to the graph DB
			Figure out what host this code is running on

		:params: params - dictonary of important configuration, right now, only the graph URI connection parameters are needed.
		"""

		self.params=params

		# Connect to graph DB
		print ("Connecting to:", params['GRAPH_URI'])
		self._conn = Neo4jConnection(uri=params['GRAPH_URI'], user=params['GRAPH_USER'], pwd=params['GRAPH_PASSWORD'])
		print ("Connection successfull:", self._conn.test_connection())
		self._host = socket.gethostname() # portable to *docker* containers
		print ("Running on:", self._host)
	
	def setNamespace(self, namespace_id: str):
		"""
		Sets the namespace for this container's commits

		:params: namespace_id - namespace value 
		"""
		self._namespace_id = namespace_id
		return self
	
	def lookupAndAttach(self, container_id):
		"""
		Checks if the node referenced by container_id is a valid container, queries the metastore for relevant metadata

		:params: container_id - the unique container ID, either as an integer (neo4j autopopulated ID) or as a string (the Qualified Path)
		"""
		self._attached = False
		print ("Lookup ID:", container_id)

		# Figure out how to match the node
		if type(container_id) is str: 
			self._match_clause = f"""WHERE container.QualifiedPath = '{container_id}'"""
		elif type(container_id) is int:
			self._match_clause = f"""WHERE id(container) = {container_id} """
		else:
			raise RuntimeError("Invalid container_id type not (str, int)")

		# Run query
		res = self._conn.query(f"""
			MATCH (container) {self._match_clause}
			RETURN id(container), labels(container), container.type, container.name, container.Namespace, container.QualifiedPath"""
		)
		
		# Check if the results are singleton (they should be... since we only query unique IDs!!!) 
		if res is None or len(res) == 0: 
			print ("Not found")
			return self

		# Set some potentially import parameters
		print ("Found:", res)
		self._container_id  = res[0]["id(container)"]
		self._name 			= res[0]["container.name"]
		self._qualifiedpath = res[0]["container.QualifiedPath"]
		self._type		 	= res[0]["container.type"]
		self._labels		= res[0]["labels(container)"]
		self._node_commits	= {}

		# Containers need to have a qualified path
		if self._qualifiedpath is None: 
			print ("Found, however not valid container object, containers must have a name, namespace, and qualified path")
			return self
		
		# Set match clause to id
		self._match_clause = f"""WHERE id(container) = {self._container_id} """
		print ("Match on:", self._match_clause)

		# Let us know attaching was a success! :)
		print ("Successfully attached to:", self._type, self._qualifiedpath)
		self._attached = True

		return self
	
	def isAttached(self):
		"""
		Returns true if container was properly attached (i.e. checks in lookupAndAttach succeeded), else False
		"""
		print ("Attached:", self._attached)

	def get(self, type, view=""):
		"""
		Query graph DB container node for dependent data nodes.  
		Parses the path field URL for various cases, and sets the node.path attribute with a corrected path
		Note: namespace is not a default filter for get nodes, but is for adding them (i.e., one can write data under a different namespace)

		:params: type - the type of data designed 
			e.g. radiomics, mha, dicom, png, svs, geojson, etc.
		:params: view - can be used to filter nodes
			e.g. data.source='generateMHD'
			e.g. data.label='Right'
			e.g. data.namespace in ['default', 'my_cohort']

		:example: get("mhd", "generate-mhd") gets data of type "mhd" generated from the source "generate-mhd"
		"""

		# Prepend AND since the query runs with a WHERE on the container ID by default
		if view is not "": view = "AND " + view

		# Run query, subject to SQL injection attacks (but right now, our entire system is)
		res = self._conn.query(f"""
			MATCH (container)-[:HAS_DATA]-(data:{type}) 
			{self._match_clause}
			{view}
			RETURN data"""
		)
		# Catches bad queries
		# If successfull query, reconstruct a Node object
		if res is None or len(res) != 1: 
			return None
		else:
			node = Node(res[0]['data']['type'], res[0]['data']['name'], dict(res[0]['data'].items()))
		
		# Parse path (filepath) URI: more sophistic path logic to come (pulling from S3, external mounts, etc!!!!)
		# For instance, if it was like s3://bucket/test.dcm, this should pull the dicom to a temporary directory and set .path to that dir
		# This also might be dependent on filetype
		# Checks:
			# If like file:/path/to.dcm, strip file:
			# Else, just return path
		# TODO: change 'path' field to 'filepath_uri"?
		if "path" in node.properties.keys(): 
			path = node.properties["path"]
			if path.split(":")[0] == "file":
				node.path = path.split(":")[-1]
			else:
				node.path = path

		# Check that we got it right, and this path is readable on the host system
		print ("Filepath is valid:", os.path.exists(node.path))
		return node
	
	def add(self, node: Node):
		"""
		Adds a node to a temporary dictonary that will be used to save/commit nodes to the relevant databases
		If you add the same node under the same name, no change as the 
		Decorates the node with the container's namespace

		:param: node - node object
		"""
		assert isinstance(node, Node)

		# Add to node commit dictonary
		print ("Adding:", node.name)
		self._node_commits[node.name] = node
		
		# Decorate with the container namespace 
		self._node_commits[node.name].properties['Namespace'] = self._namespace_id
		print ("Container has {0} pending commits".format(len(self._node_commits)))

	def saveAll(self):
		"""
		Tries to create nodes for all committed nodes
		"""
		# Loop through all nodes in commit dictonary, and run query
		for n in self._node_commits.values():
			self._conn.query(f""" 
				MATCH (container) {self._match_clause}
				MERGE (da:{n.get_create_str()})
				MERGE (container)-[:HAS_DATA]->(da)"""
			)

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

