import re


def dms_to_dd(input_string: str) -> float:
    """
    Converts string value of degree minute seconds into a decimal degree float.

    :param input_string: string following the form "degree minute' seconds"
    :return: decimal degree float
    """
    input_string += " "
    try:
        degrees = re.search(r"[\d\.]+ ", input_string).group(0).replace(" ", "")
    except AttributeError:
        degrees = 0
    try:
        minutes = re.search(r"[\d\.]+\'", input_string).group(0).replace("\'", "")
    except AttributeError:
        minutes = 0
    try:
        seconds = re.search(r"[\d\.]+\"", input_string).group(0).replace("\"", "")
    except AttributeError:
        seconds = 0
    return float(degrees) + float(minutes) / 60 + float(seconds) / 3600
