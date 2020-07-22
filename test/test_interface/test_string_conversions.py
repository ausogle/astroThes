from src.interface.string_conversions import dms_to_dd
import pytest


@pytest.mark.parametrize("input_string, expected", [("1 30' 3600\"", 2.5),
                                                    ("0 0' 0\"", 0)])
def test_dms_to_dd(input_string, expected):
    actual = dms_to_dd(input_string)
    assert actual == expected

