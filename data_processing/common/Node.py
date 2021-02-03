from data_processing.common.utils import to_sql_field, does_not_contain
import warnings

class Node(object):
	"""
	Node object defines the type and attributes of a graph node.

	:param: node_type: node type. e.g. scan
	:param: name: required node name. e.g. scan-123
	:param: properties: dict of key, value pairs for the node.
	"""
	def __init__(self, node_type, node_name, properties={}):

		# Required schema: node_type, node_name

		self.type = node_type
		self.name = node_name
		self.properties = properties

		if self.type=="cohort":
			self.properties['Namespace'] = node_name

		if not "Namespace" in properties.keys():
			self.properties['Namespace'] = 'default'

		self.properties["QualifiedPath"] = self.get_qualified_name(self.properties['Namespace'], self.name)
		self.properties["type"] = self.type

	def set_namespace(self, namespace_id: str):
		"""
		Sets the namespace for this Node commits

		:params: namespace_id - namespace value 
		"""
		self.properties['Namespace'] = namespace_id
		self.properties["QualifiedPath"] = self.get_qualified_name(self.properties['Namespace'], self.name)

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
		Returns a string representation of the node with only the QualifiedPath as a property
		"""
		kv = self.get_all_props()

		prop_string = self.prop_str( ["QualifiedPath"], kv)
		return f"""{self.type}:globals{{ {prop_string} }}"""
	
	def get_map_str(self):
		"""
		Returns the properties as a cypher map
		"""
		kv = self.get_all_props()

		prop_string = self.prop_str(kv.keys(), kv)
		return f"""{{ {prop_string} }}"""



	@staticmethod
	def prop_str(fields, row):
		"""
		Returns a kv string like 'id: 123, ...' where prop values come from row.
		"""
		fields = set(fields).intersection(set(row.keys()))

		kv = [f" {to_sql_field(x)}: '{row[x]}'" for x in fields]
		return ','.join(kv)
	
	@staticmethod
	def prop_str_repr(fields, row):
		"""
		Returns a kv string like ' - id: 123 <newline> ...' where prop values come from row.
		"""
		fields = set(fields).intersection(set(row.keys()))

		kv = [f"   {to_sql_field(x)}: '{row[x]}'" for x in fields]
		return '\n'.join(kv)
	
	@staticmethod
	def get_qualified_name(namespace, identifier): 
		"""
		Returns the full name given a namespace and patient ID
		"""
		does_not_contain(":", namespace)
		does_not_contain(":", identifier)

		return f"{namespace}::{identifier}"
	
