from data_processing.common.custom_logger import init_logger
from data_processing.common.Neo4jConnection import Neo4jConnection
import os, socket


class Container(object):
	"""
	Container: an abstraction with an id, name, namespace, type, and a list of associated data nodes

	Interfaces with a metadata store (graph DB) and raw file stores (gpfs, potentially others)

	Handles the matching and creation of metadata

	Example usage:
	$ container = data_processing.common.GraphEnum.Container( params ).setNamespace("test").lookupAndAttach("1.2.840...")
		> Connecting to: neo4j://localhost:7687
		> Connection successfull: True
		> Running on: localhost
		> Lookup ID: 1.2.840...
		> Found: [<Record id(container)=7091 labels(container)=['scan'] container.type='scan' container.name='1.2.840...>]
		> Match on: WHERE id(container) = 7091 
		> Successfully attached to: scan 1.2.840...

	$ node = Node("dicom", "DCM-0123", {"Modality":"CT", "path":"file:/some/path/1.dcm"})

	$ container.add(node)
		> Adding: test-0000
		  Container has 1 pending commits

	$ container.saveAll()
		> Committing dicom:globals{  name: 'DCM-0123', QualifiedPath: 'test::DCM-0123', Namespace: 'test', type: 'dicom' , path: 'file:/some/path/1.dcm'}

	$ node = container.ls("dicom")
		> ----------------------------------------------------------------------------------------------------
		  name: DCM-0123
		  type: dicom
		  properties: 
		  - type: 'dicom'
		  - QualifiedPath: 'test::DCM-0123'
		  - path: 'file:/some/path/1.dcm'
		  - Namespace: '3'
		  - Modality: 'CT'
		  - name: 'DCM-0123'
		  ----------------------------------------------------------------------------------------------------
	$ container.get("dicom", "data.Namespace='test'").path
		> /some/path/1.dcm

	$ container.get("dicom", "data.Namespace='test'").properties['Modality']
		> 'CT'

	
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

		self.logger = init_logger("common-container.log", "Container [empty]")

		self.params=params

		# Connect to graph DB
		self.logger.info ("Connecting to: %s", params['GRAPH_URI'])
		self._conn = Neo4jConnection(uri=params['GRAPH_URI'], user=params['GRAPH_USER'], pwd=params['GRAPH_PASSWORD'])
		self.logger.info ("Connection test: %s", self._conn.test_connection())
		self._host = socket.gethostname() # portable to *docker* containers
		self.logger.info ("Running on: %s", self._host)
	
	def setNamespace(self, namespace_id: str):
		"""
		Sets the namespace for this container's commits

		:params: namespace_id - namespace value 
		"""
		self._namespace_id = namespace_id
		self.logger.info ("Container namespace: %s", self._namespace_id)

		return self
	
	def lookupAndAttach(self, container_id):
		"""
		Checks if the node referenced by container_id is a valid container, queries the metastore for relevant metadata

		:params: container_id - the unique container ID, either as an integer (neo4j autopopulated ID) or as a string (the Qualified Path)
		"""
		self._attached = False
		self.logger.info ("Lookup ID: %s", container_id)

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
			self.logger.warning ("Not found")
			return self

		# Set some potentially import parameters
		self.logger.info ("Found: %s", res)
		self._container_id  = res[0]["id(container)"]
		self._name 			= res[0]["container.name"]
		self._qualifiedpath = res[0]["container.QualifiedPath"]
		self._type		 	= res[0]["container.type"]
		self._labels		= res[0]["labels(container)"]
		self._node_commits	= {}

		# Containers need to have a qualified path
		if self._qualifiedpath is None: 
			self.logger.warning ("Found, however not valid container object, containers must have a name, namespace, and qualified path")
			return self
		
		# Set match clause to id
		self._match_clause = f"""WHERE id(container) = {self._container_id} """
		self.logger.debug ("Match on: %s", self._match_clause)

		# Let us know attaching was a success! :)
		self.logger = init_logger("common-container.log", f"Container [{self._container_id}]")
		self.logger.info ("Successfully attached to: %s %s", self._type, self._qualifiedpath)
		self._attached = True

		return self
	
	def isAttached(self):
		"""
		Returns true if container was properly attached (i.e. checks in lookupAndAttach succeeded), else False
		"""
		self.logger.info ("Attached: %s", self._attached)

	def ls(self, type, view=""):
		"""
		Query graph DB container node for dependent data nodes, and list them  

		:params: type - the type of data designed 
			e.g. radiomics, mha, dicom, png, svs, geojson, etc.
		:params: view - can be used to filter nodes
			e.g. data.source='generateMHD'
			e.g. data.label='Right'
			e.g. data.namespace in ['default', 'my_cohort']

		:example: ls("png") gets data nodes of type "png" and prints the repr of each node
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
		if res is None or len(res) == 0: 
			return None
		else:
			[self.logger.info(Node(rec['data']['type'], rec['data']['name'], dict(rec['data'].items()))) for rec in res]


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

		:example: get("mhd", "data.MethodID = 'generate-mhd'") gets data of type "mhd" generated from the method "generate-mhd"
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
		
			# Output and check
			self.logger.info ("Resolved %s -> %s", node.properties["path"], node.path)

			# Check that we got it right, and this path is readable on the host system
			self.logger.info ("Filepath is valid: %s", os.path.exists(node.path))

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
		self.logger.info ("Adding: %s", node.name)
		self._node_commits[node.name] = node
		
		# Decorate with the container namespace 
		self._node_commits[node.name].set_namespace( self._namespace_id )
		self.logger.info ("Container has %s pending commits",  len(self._node_commits))

	def saveAll(self):
		"""
		Tries to create nodes for all committed nodes
		"""
		# Loop through all nodes in commit dictonary, and run query
		for n in self._node_commits.values():
			self.logger.info ("Committing %s", n.get_create_str())
			self._conn.query(f""" 
				MATCH (container) {self._match_clause}
				MERGE (da:{n.get_create_str()})
				MERGE (container)-[:HAS_DATA]->(da)"""
			)