import os
import pytest

from data_processing.common.DataStore import DataStore_v2

@pytest.fixture()
def datastore2():
    os.environ['LUNA_HOME'] = 'tests/data_processing/common/testdata/'
    datastore2 = DataStore_v2('tests/data_processing/common/testdata/store')
    yield datastore2

def test_datastore_v2_init():
    os.environ['LUNA_HOME'] = ''
    with pytest.raises(RuntimeError):
        DataStore_v2('tests/data_processing/common/testdata/store')
