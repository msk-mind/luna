from luna.common.utils import to_sql_field, to_sql_value, does_not_contain
import warnings, os
from pathlib import Path

# TODO: Update types to match docs
CONTAINER_TYPES = ["cohort", "patient", "scan", "slide", "parquet", "accession", "generic"]
RADIOLOGY_TYPES = ["DicomSeries", "DicomImageSeries", "DicomImage", "VolumetricImage", "RadiologyScan", "VolumetricLabel", "VolumetricLabelSet", "Voxels", "Radiomics"]
PATHOLOGY_TYPES = ["WholeSlideImage", "WsiThumbnail", "TileScores", "TileImages", "PointAnnotation", "PointAnnotationJson", "RegionalAnnotationBitmap", "RegionalAnnotationJSON", "CellMap"]
GENERIC_TYPES   = ["ResultSegment"]
ALL_DATA_TYPES  = RADIOLOGY_TYPES + PATHOLOGY_TYPES + GENERIC_TYPES

class Node(object):
	"""
	Node object defines the type and attributes of a graph node.

	:param: node_type: node type. e.g. scan
	:param: name: required node name. e.g. scan-123
	:param: properties: dict of key, value pairs for the node.
	"""
	def __init__(self, node_type, node_name, properties=None):

		# Required schema: node_type, node_name

		self.type = node_type

		if properties is None:
			self.properties = {}
		else:
			self.properties = properties

		# Special case: a cohort name is it's own namespace
		if self.type in ["cohort"]:
			self.properties['namespace'] = node_name

		# For containers, DB name is the name, for data type nodes, it's type-name, and must be one of these two categories
		if self.type in CONTAINER_TYPES:
			self.name = f'{node_name}'
		elif self.type in ALL_DATA_TYPES:
			self.name = f'{node_type}-{node_name}'
		else:
			raise RuntimeError(f"Invalid Node Data Type {self.type}")


		if   "namespace" in self.properties.keys() and "subspace" in self.properties.keys():
                        self.properties["qualified_address"] = self.get_qualified_name(self.properties['namespace'], self.properties['subspace'], self.name)
		elif "namespace" in self.properties.keys():
			self.properties["qualified_address"] = self.get_qualified_name(self.properties['namespace'], self.name)
		else:
			self.properties['qualified_address'] = self.name

		self.properties["type"] = self.type

		self.set_data( self.properties.get("data", None))
		self.set_aux ( self.properties.get("aux",  None))

	def set_namespace(self, namespace_id: str, subspace_id=None):
		"""
		Sets the namespace for this Node commits

		:params: namespace_id - namespace value
		:params: subspace_id  - subspace value, optional
		"""
		self.properties['namespace'] = namespace_id

		if subspace_id is None:
			self.properties["qualified_address"] = self.get_qualified_name(self.properties['namespace'], self.name)
		else:
			self.properties['subspace']  = subspace_id
			self.properties["qualified_address"] = self.get_qualified_name(self.properties['namespace'], self.properties['subspace'], self.name)


	def set_data(self, data):
		self.data = None

		if data is None: return

		if isinstance(data, str):
			data = Path(data)

		if not isinstance(data, Path):
			raise RuntimeError("set_data() only accepts path-like objects")

		if not data.exists():
			raise RuntimeError("Tried to set data to a non-existent path!")

		self.properties['data'] = str(data)
		self.data = str(data)

	def set_aux(self, aux):
		self.aux = None

		if aux is None: return

		if isinstance(aux, str):
			aux = Path(aux)

		if not isinstance(aux, Path):
			raise RuntimeError("set_aux() only accepts path-like objects")

		if not aux.exists():
			raise RuntimeError("Tried to set aux to a non-existent path!")

		self.properties['aux'] = str(aux)
		self.aux = str(aux)

	def __repr__(self):
		"""
		Returns a string representation of the node
		"""
		kv = self.get_all_props()
		prop_string = self.prop_str_repr(kv.keys(), kv)
		bigline = "-"*100
		return f"{bigline}\nname: {self.name}\ntype: {self.type}\nproperties: \n{prop_string}\n{bigline}"

	def get_all_props(self):
		"""
		Name is a required field, but it's still a property of this node.
		Return the properties as a dict including the name property!
		"""
		kv = self.properties
		kv["name"] = self.name

		return kv

	def get_create_str(self):
		"""
		Returns a string representation of the node with all properties
		"""
		kv = self.get_all_props()

		prop_string = self.prop_str(kv.keys(), kv)
		return f"""{self.type}:globals{{ {prop_string} }}"""

	def get_match_str(self):
		"""
		Returns a string representation of the node with only the qualified_address as a property
		"""
		kv = self.get_all_props()

		prop_string = self.prop_str( ["qualified_address"], kv)
		return f"""{self.type}:globals{{ {prop_string} }}"""

	def get_map_str(self):
		"""
		Returns the properties as a cypher map
		"""
		kv = self.get_all_props()

		prop_string = self.prop_str(kv.keys(), kv)
		return f"""{{ {prop_string} }}"""

	def get_address(self):
		"""
		Returns current node address
		"""
		return self.properties.get("qualified_address")

	@staticmethod
	def prop_str(fields, row):
		"""
		Returns a kv string like 'id: 123, ...' where prop values come from row.
		"""
		fields = set(fields).intersection(set(row.keys()))

		kv = [f" {to_sql_field(x)}: {to_sql_value(row[x])}" for x in fields]
		return ','.join(kv)

	@staticmethod
	def prop_str_repr(fields, row):
		"""
		Returns a kv string like ' - id: 123 <newline> ...' where prop values come from row.
		"""
		fields = set(fields).intersection(set(row.keys()))

		kv = [f"   {to_sql_field(x)}: {to_sql_value(row[x])}" for x in fields]
		return '\n'.join(kv)

	@staticmethod
	def get_qualified_name(*args):
		"""
		Returns the full name given a namespace and patient ID
		"""
		for name in args: does_not_contain(":", name)

		return "::".join(args)

