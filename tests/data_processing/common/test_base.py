import pytest
import os
def test_os():
    my_var = os.environ['ENV_DOES_NOT_EXIST']
    assert my_var == "foo"



