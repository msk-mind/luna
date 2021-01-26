from filehash import FileHash
from io import BytesIO
import os, json

def to_sql_field(s):
	filter1 = s.replace(".","_").replace(" ","_")
	filter2 = ''.join(e for e in filter1 if e.isalnum())
	return filter2

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
	:param prefix: e.g. DICOM-
	:return: string uuid
	"""
	posix_file_path = path.split(':')[-1]

	rec_hash = FileHash('sha256').hash_file(posix_file_path)
	return prefix + rec_hash


def generate_uuid_binary(content, prefix):
	"""
	Returns hash of the binary, preceded by the prefix.
	:param content: binary
	:param prefix: e.g. FEATURE-
	:return: string uuid
	"""
	content = BytesIO(content)

	import EnsureByteContext
	with EnsureByteContext.EnsureByteContext():
		uuid = FileHash('sha256').hash_file(content)

	return prefix + uuid


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

    method_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data/COHORTS", cohort_id, "methods")
    with open(os.path.join(method_dir, f'{method_id}.json')) as json_file:
        method_config = json.load(json_file)['params']
    return method_config

