from filehash import FileHash
from io import BytesIO
import os, json, yaml
import logging
logger = logging.getLogger(__name__)

def to_sql_field(s):
	filter1 = s.replace(".","_").replace(" ","_")
	filter2 = ''.join(e for e in filter1 if e.isalnum() or e=='_')
	return filter2

def to_sql_value(s):
	if isinstance(s, str): return f"'{s}'"
	if not s==s:  return 'Null'
	if s is None: return 'Null'
	else: return f"{s}"


def clean_nested_colname(s):
	"""
	Removes map name for MapType columns.
	e.g. metadata.SeriesInstanceUID -> SeriesInstanceUID
	"""
	return s[s.find('.')+1:]

def generate_uuid(path, prefix):
	"""
	Returns hash of the file given path, preceded by the prefix.
	:param path: file path e.g. file:/path/to/file
	:param prefix: list e.g. ["SVGEOJSON","default-label"]
	:return: string uuid
	"""
	posix_file_path = path.split(':')[-1]

	rec_hash = FileHash('sha256').hash_file(posix_file_path)
	prefix.append(rec_hash)
	return "-".join(prefix)


def generate_uuid_binary(content, prefix):
	"""
	Returns hash of the binary, preceded by the prefix.
	:param content: binary
	:param prefix: list e.g. ["FEATURE"]
	:return: string uuid
	"""
	content = BytesIO(content)

	import EnsureByteContext
	with EnsureByteContext.EnsureByteContext():
		uuid = FileHash('sha256').hash_file(content)

	prefix.append(uuid)
	return "-".join(prefix)


def generate_uuid_dict(json_str, prefix):
	"""
	Returns hash of the json string, preceded by the prefix.
	:param json_str: str representation of json
	:param prefix: list e.g. ["SVGEOJSON","default-label"]
	:return: v
	"""
	json_bytes = json.dumps(json_str).encode('utf-8')

	import EnsureByteContext
	with EnsureByteContext.EnsureByteContext():
		uuid = FileHash('sha256').hash_file(BytesIO(json_bytes))

	prefix.append(uuid)
	return "-".join(prefix)

def does_not_contain(token, value):
	"""
	Validate that `token` is not a substring of `value`

	:param: token: string e.g. : | .
	:param: value: dictionary, list, or str
	"""
	if isinstance(value, str):
		if token in value:
			raise ValueError(f"{value} cannot contain {token}")

	if isinstance(value, list):
		if any([token in v for v in value]):
			raise ValueError(str(value) + f" cannot contain {token}")

	if isinstance(value, dict):
		if any([isinstance(key, str) and token in key or isinstance(val, str) and token in val for key,val in value.items()]):
			raise ValueError(str(value) + f" cannot contain {token}")

	return True


def replace_token(token, token_replacement, value):
	"""
	Replace `token` with `token_replacement` in `value`

	:param: token: string e.g. : | .
	:param: token_replacement: string e.g. _ -
	:param: value: dictionary, list, or str
	"""
	if isinstance(value, str):
		return value.replace(token, token_replacement)

	if isinstance(value, list):
		new_value = []
		for v in value:
			new_value.append(v.replace(token, token_replacement))
		return new_value

	if isinstance(value, dict):
		new_value = {}
		for key,val in value.items():
			new_key, new_val = key, val
			if isinstance(key, str):
				new_key = key.replace(token, token_replacement)
			if isinstance(val, str):
				new_val = val.replace(token, token_replacement)
			new_value[new_key] = new_val

		return new_value

	return value


def get_method_data(cohort_id, method_id):
	"""
	Return method dict

	:param: cohort_id: string
	:param: method_id: string
	"""

	method_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", cohort_id, "methods")
	with open(os.path.join(method_dir, f'{method_id}.json')) as json_file:
		method_config = json.load(json_file)['params']
	return method_config

def get_absolute_path(module_path, relative_path):
	""" Given the path to a module file and the path, relative to the module file, of another file
	that needs to be referenced in the module, this method returns the absolute path of the file
	that needs to be referenced.

	This method makes it possible to resolve absolute paths to files in any environment a
	module and the referenced files are deployed to.

	:param module_path path to the module. Use '__file__' from the module.
	:param relative_path path to the file that needs to be referenced by the module. The path must
	be relative to the module.
	:return absolute path to file with the specified relative_path
	"""
	path = os.path.join(os.path.dirname(module_path), relative_path)

	# resolve any back-paths with ../ to simplify absolute path
	return os.path.realpath(path)

def validate_params(given_params, params_list):
	logger = logging.getLogger(__name__)

	d_params = {}
	for name, dtype in params_list:
		if given_params.get(name, None) == None: 
			raise RuntimeError(f"Param {name} of type {type} was never set, but required by transform, please check your input variables.")
		try:
			if 'List' in str(dtype):
				if type (given_params[name])==list:
					d_params[name] = given_params[name]
				else:
					d_params[name] = [dtype.__args__[0](s) for s in given_params[name].split(',')]
			elif dtype==dict:
				if type (given_params[name])==dict:
					d_params[name] = given_params[name]
				else:
					d_params[name] = {s.split('=')[0] : s.split('=')[1] for s in given_params[name].split(',')}
			elif dtype==json:
				if type (given_params[name])==dict:
					d_params[name] = given_params[name]
				else:
					d_params[name] = json.loads(given_params[name])
			elif type(dtype)==type:
				d_params[name] = dtype(given_params[name])
			else:
				raise RuntimeError(f"Type {type(dtype)} invalid!")
		except ValueError:
			raise RuntimeError(f"Param {name} could not be cast to {dtype}")
		except RuntimeError as e:
			raise e
		logger.info (f"Param {name} set = {d_params[name]}")
	return d_params

def expand_inputs(given_params):
	d_params = {}

	for key, value in given_params.items():
		if 'input_' in key:
			# For some inputs, they may be implicit, where metadata about them is at the provided directory path
			expected_metadata = os.path.join(value, 'metadata.json')

			if os.path.isdir(value) and os.path.exists(expected_metadata):
				# We supplied an inferred input from some previous output, so figure it out from the metadata of this output fold

				with open(expected_metadata, 'r') as yaml_file:
					metadata = yaml.safe_load(yaml_file)

				# Output names/slots are same as input names/slots, just without input_ prefix
				expanded_input = metadata.get ( key.replace('input_', ''), None)

				# Tree output_directories should never be passed to functions which cannot accept them
				if expanded_input is None:
					raise RuntimeError (f"No matching output slot of type [{key.replace('input_', '')}] at given input directory")

				logger.info(f"Expanded input {value} -> {expanded_input}")

				d_params[key] = expanded_input
			else:
				d_params[key] = value
		else:
			d_params[key] = value

	return d_params


def cli_runner(cli_kwargs, cli_params, cli_function):
	logger.info ("Running {cli_function} with {cli_kwargs}")
	kwargs = {}
	if not 'output_dir' in cli_kwargs.keys():
		raise RuntimeError("CLI Runners assume an output directory")

	# Get params from param file
	if cli_kwargs.get('method_param_path'):
		with open(cli_kwargs.get('method_param_path'), 'r') as yaml_file:
			yaml_kwargs = yaml.safe_load(yaml_file)
		kwargs.update(yaml_kwargs) # Fill from json
	
	for key in list(cli_kwargs.keys()):
		if cli_kwargs[key] is None: del cli_kwargs[key]

	# Override with CLI arguments
	kwargs.update(cli_kwargs) # 
	
	# Validate them
	kwargs = validate_params(kwargs, cli_params)

	# Expand implied inputs
	kwargs = expand_inputs(kwargs)

	output_dir = kwargs['output_dir']
	os.makedirs(output_dir, exist_ok=True)

	result = cli_function(**kwargs)

	kwargs.update(result)

	with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
		json.dump(kwargs, fp, indent=4, sort_keys=True)