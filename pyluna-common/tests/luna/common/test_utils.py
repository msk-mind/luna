from luna.common.utils import (
    generate_uuid,
    get_absolute_path,
    validate_params,
    cli_runner,
    expand_inputs,
)
import pytest
import os
import yaml


def test_generate_uuid():
    uuid = generate_uuid(
        "file:./pyluna-common/tests/luna/common/test_config.yml", ["FEATURE", "label"]
    )

    assert uuid.startswith("FEATURE-label-")


def test_get_absolute_path():
    absolute_path = get_absolute_path(
        __file__, "../data_ingestion_template_invalid.yml"
    )
    assert absolute_path.startswith("/")
    assert absolute_path.endswith(
        "pyluna-common/tests/luna/data_ingestion_template_invalid.yml"
    )


def test_validate_params__simple():
    params_list = [("input", str), ("threshold", float)]

    given_params = {
        "input": "data.json",
        "threshold": 0.5,
        "extra": 1,
    }
    out_params = validate_params(given_params, params_list)

    assert out_params == {
        "input": "data.json",
        "threshold": 0.5,
    }


def test_validate_params__casting():
    params_list = [("input", str), ("threshold", float)]

    given_params = {
        "input": "data.json",
        "threshold": "0.5",  # Cast string
        "extra": 1,
    }
    out_params = validate_params(given_params, params_list)

    assert out_params == {
        "input": "data.json",
        "threshold": 0.5,
    }


def test_validate_params__missing_value():
    with pytest.raises(RuntimeError):

        params_list = [("input", str), ("threshold", float)]

        given_params = {
            "threshold": 0.5,  # Cast string
            "extra": 1,
        }
        out_params = validate_params(given_params, params_list)

        assert out_params == {
            "input": "data.json",
            "threshold": 0.5,
        }


def test_validate_params__wrong_value():
    with pytest.raises(RuntimeError):

        params_list = [("input", str), ("threshold", float)]

        given_params = {
            "input": "data.json",
            "threshold": "Five",  # Cast string
            "extra": 1,
        }
        out_params = validate_params(given_params, params_list)

        assert out_params == {
            "input": "data.json",
            "threshold": 0.5,
        }


def test_validate_params_dict():
    params_list = [("input1", dict), ("input2", dict)]

    given_params = {
        "input1": "{'key':'not_json_format'}",
        "input2": {"key": "not_json_format"},
    }

    out_params = validate_params(given_params, params_list)

    assert out_params == {
        "input1": {"key": "not_json_format"},
        "input2": {"key": "not_json_format"},
    }


def test_expand_inputs_explicit():
    given_params = {
        "input_text_file": "pyluna-common/tests/luna/common/testdata/simple_output_directory/result.txt",
        "threshold": "Five",
        "extra": 1,
    }

    kwargs, keys = expand_inputs(given_params)

    assert (
        kwargs["input_text_file"]
        == "pyluna-common/tests/luna/common/testdata/simple_output_directory/result.txt"
    )


def test_expand_inputs_implicit():

    given_params = {
        "input_text_file": "pyluna-common/tests/luna/common/testdata/simple_output_directory/",
        "threshold": "Five",
        "extra": 1,
    }

    kwargs, keys = expand_inputs(given_params)

    assert (
        kwargs["input_text_file"]
        == "pyluna-common/tests/luna/common/testdata/simple_output_directory/result.txt"
    )


def test_expand_inputs_implicit_but_missing():
    with pytest.raises(RuntimeError):
        given_params = {
            "input_csv_file": "pyluna-common/tests/luna/common/testdata/simple_output_directory/",
            "threshold": "Five",
            "extra": 1,
        }

        expand_inputs(given_params)


def test_cli_runner(tmp_path):
    def simple_transform(input_text_file, message, output_dir, username):
        with open(input_text_file, "r") as fp:
            old_message = fp.read()

        new_message = old_message + "\n" + message
        with open(output_dir + "/message.txt", "w") as fp:
            fp.write(new_message)

        properties = {
            "num_characters": len(new_message),
        }

        return properties

    cli_kwargs = {
        "input_text_file": "pyluna-common/tests/luna/common/testdata/simple_output_directory/result.txt",
        "output_dir": tmp_path,
        "message": "Hello to you too!",
        "username": "my-user",
    }

    _params_ = [
        ("input_text_file", str),
        ("output_dir", str),
        ("message", str),
        ("username", str),
    ]
    cli_runner(cli_kwargs, _params_, simple_transform)

    assert os.path.exists(str(tmp_path) + "/metadata.yml")
    assert os.path.exists(str(tmp_path) + "/message.txt")

    with open(str(tmp_path) + "/metadata.yml") as fp:
        metadata = yaml.safe_load(fp)
        assert metadata["num_characters"] == 24


def test_cli_runner_reruns(tmp_path):
    def simple_transform(input_text_file, message, output_dir):
        with open(input_text_file, "r") as fp:
            old_message = fp.read()

        new_message = old_message + "\n" + message
        with open(output_dir + "/message.txt", "w") as fp:
            fp.write(new_message)

        properties = {
            "num_characters": len(new_message),
        }

        return properties

    cli_kwargs = {
        "input_text_file": "pyluna-common/tests/luna/common/testdata/simple_output_directory/result.txt",
        "output_dir": tmp_path,
        "message": "Hello to you too!",
    }

    _params_ = [("input_text_file", str), ("output_dir", str), ("message", str)]
    cli_runner(cli_kwargs, _params_, simple_transform)

    cli_kwargs_rerun = {
        "method_param_path": str(tmp_path) + "/metadata.yml",
        "output_dir": str(tmp_path) + "rerun",
    }

    cli_runner(cli_kwargs_rerun, _params_, simple_transform)

    with open(str(tmp_path) + "rerun" + "/metadata.yml") as fp:
        metadata = yaml.safe_load(fp)
        assert metadata["num_characters"] == 24
