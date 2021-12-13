from luna.common.utils import *
import sys
import pytest

def test_generate_uuid():
    uuid = generate_uuid("file:./pyluna-common/tests/luna/common/test_config.yml", ["FEATURE", "label"])

    assert uuid.startswith("FEATURE-label-")

def test_get_absolute_path():
    absolute_path= get_absolute_path(__file__, '../data_ingestion_template_invalid.yml')
    assert absolute_path.startswith('/')
    assert absolute_path.endswith('pyluna-common/tests/luna/data_ingestion_template_invalid.yml')



def test_validate_params__simple():
    params_list = [('input',str), ('threshold', float)]
    
    given_params = {
        'input':'data.json',
        'threshold':0.5,
        'extra':1,
    }
    out_params = validate_params(given_params, params_list)

    assert out_params == {
        'input':'data.json',
        'threshold':0.5,
    }

def test_validate_params__casting():
    params_list = [('input',str), ('threshold', float)]
    
    given_params = {
        'input':'data.json',
        'threshold':'0.5', # Cast string
        'extra':1,
    }
    out_params = validate_params(given_params, params_list)

    assert out_params == {
        'input':'data.json',
        'threshold':0.5,
    }

def test_validate_params__missing_value():
    with pytest.raises(RuntimeError):

        params_list = [('input',str), ('threshold', float)]
        
        given_params = {
            'threshold':0.5, # Cast string
            'extra':1,
        }
        out_params = validate_params(given_params, params_list)

        assert out_params == {
            'input':'data.json',
            'threshold':0.5,
        }

def test_validate_params__wrong_value():
    with pytest.raises(RuntimeError):

        params_list = [('input',str), ('threshold', float)]
        
        given_params = {
            'input':'data.json',
            'threshold':'Five', # Cast string
            'extra':1,
        }
        out_params = validate_params(given_params, params_list)

        assert out_params == {
            'input':'data.json',
            'threshold':0.5,
        }