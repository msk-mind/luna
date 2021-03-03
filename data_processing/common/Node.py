from data_processing.common.utils import to_sql_field, does_not_contain
import warnings, os

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
		self.name = node_name
                
		if properties is None: 
			self.properties = {}
		else: 
			self.properties = properties 

		if self.type=="cohort":
			self.properties['namespace'] = node_name

		if not "namespace" in self.properties.keys():
			self.properties['namespace'] = 'default'

		self.properties["qualified_address"] = self.get_qualified_name(self.properties['namespace'], self.name)
		self.properties["type"] = self.type

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
	def get_qualified_name(*args): 
		"""
		Returns the full name given a namespace and patient ID
		"""
		for name in args: does_not_contain(":", name)

		return "::".join(args).lower()
	
