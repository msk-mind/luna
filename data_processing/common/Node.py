
class Node(object):
	"""
	Node object defines the type and attributes of a graph node.

	:param: node_type: node type. e.g. scan
	:param: properties: dict of key, value pairs for the node.
	"""
	def __init__(self, node_type, node_id, properties={}):

		self.type = node_type
		self.id = node_id
		self.properties = properties

		if self.type=="cohort":
			if not "CohortID" in properties.keys():
				raise RuntimeError("Cohorts must have a CohortID!")
			self.properties["QualifiedPath"] = self.get_qualified_name(properties["CohortID"], properties["CohortID"])

		if self.type=="patient":
			if not ("PatientID" in properties.keys() and "Namespace" in properties.keys()):
				raise RuntimeError("Patients must have a PatientID and Namespace property!")
			self.properties["QualifiedPath"] = self.get_qualified_name(properties["Namespace"], properties["PatientID"])
		
		if self.type in ["dicom", "mha", "mhd", "radiomics"]: # TODO pull from a config/DB of sorts
			if not ("RecordID" in properties.keys() and "Namespace" in properties.keys() ):
				raise RuntimeError("metadata must have a RecordID, Namespace!")
			self.properties["QualifiedPath"] = self.get_qualified_name(properties["Namespace"], properties["RecordID"])
		
		if self.type=="method":
			if not ("MethodID" in properties.keys() and "Namespace" in properties.keys()):
				raise RuntimeError("method must have a MethodID and Namespace")
			self.properties["QualifiedPath"] = self.get_qualified_name(properties["Namespace"], properties["MethodID"])	

	def create(self):
		prop_string = self.prop_str(self.properties.keys(), self.properties)
		return f"""{self.type}:globals{{ {prop_string} }}"""

	def match(self):
		prop_string = self.prop_str( ["QualifiedPath"], self.properties)
		return f"""{self.type}{{ {prop_string} }}"""

	@staticmethod
	def prop_str(fields, row):
		"""
		Returns a kv string like 'id: 123, ...' where prop values come from row.
		"""
		fields = set(fields).intersection(set(row.keys()))

		kv = [f" {x}: '{row[x]}'" for x in fields]
		return ','.join(kv)
	
	@staticmethod
	def get_qualified_name(namespace, identifier): 
		"""
		Returns the full name given a namespace and patient ID
		"""
		if ":" in namespace or ":" in identifier: raise ValueError("Qualified path cannot be constructed, namespace or identifier cannot contain ':'")
		return f"{namespace}::{identifier}"
	
