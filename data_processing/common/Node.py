from data_processing.common.utils import to_sql_field, does_not_contain

class Node(object):
	"""
	Node object defines the type and attributes of a graph node.

	:param: node_type: node type. e.g. scan
	:param: name: required node name. e.g. scan-123
	:param: properties: dict of key, value pairs for the node.
	"""
	def __init__(self, node_type, name, properties={}):

		self.type = node_type
		self.name = name
		self.properties = properties

		if self.type=="cohort":
			self.properties["QualifiedPath"] = self.get_qualified_name(self.name, self.name)

		else:
			if not "Namespace" in properties.keys():
				raise RuntimeError("Missing required Namespace property!")
			self.properties["QualifiedPath"] = self.get_qualified_name(properties["Namespace"], self.name)	

	def get_all_props(self):
		"""
		Name is a required field, but it's still a property of this node.
		Return the properties including the name property!
		"""
		kv = self.properties
		kv["name"] = self.name

		return kv

	def get_create_str(self):

		kv = self.get_all_props()

		prop_string = self.prop_str(kv.keys(), kv)
		return f"""{self.type}:globals{{ {prop_string} }}"""

	def get_match_str(self):

		kv = self.get_all_props()

		prop_string = self.prop_str( ["QualifiedPath"], kv)
		return f"""{self.type}{{ {prop_string} }}"""

	@staticmethod
	def prop_str(fields, row):
		"""
		Returns a kv string like 'id: 123, ...' where prop values come from row.
		"""
		fields = set(fields).intersection(set(row.keys()))

		kv = [f" {to_sql_field(x)}: '{row[x]}'" for x in fields]
		return ','.join(kv)
	
	@staticmethod
	def get_qualified_name(namespace, identifier): 
		"""
		Returns the full name given a namespace and patient ID
		"""
		does_not_contain(":", namespace)
		does_not_contain(":", identifier)

		return f"{namespace}::{identifier}"
	