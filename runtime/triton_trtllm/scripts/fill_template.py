# /usr/bin/env python3
from argparse import ArgumentParser
from string import Template


def split(string, delimiter):
    """Split a string using delimiter. Supports escaping.

    Args:
        string (str): The string to split.
        delimiter (str): The delimiter to split the string with.

    Returns:
        list: A list of strings.
    """
    result = []
    current = ""
    escape = False
    for char in string:
        if escape:
            current += char
            escape = False
        elif char == delimiter:
            result.append(current)
            current = ""
        elif char == "\\":
            escape = True
        else:
            current += char
    result.append(current)
    return result


def main(file_path, substitutions, in_place):
    with open(file_path) as f:
        pbtxt = Template(f.read())

    sub_dict = {
        "max_queue_size": 0,
        'max_queue_delay_microseconds': 0,
    }
    for sub in split(substitutions, ","):
        key, value = split(sub, ":")
        sub_dict[key] = value

        assert key in pbtxt.template, f"key '{key}' does not exist in the file {file_path}."

    pbtxt = pbtxt.safe_substitute(sub_dict)

    if in_place:
        with open(file_path, "w") as f:
            f.write(pbtxt)
    else:
        print(pbtxt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file_path", help="path of the .pbtxt to modify")
    parser.add_argument(
        "substitutions",
        help="substitutions to perform, in the format variable_name_1:value_1,variable_name_2:value_2..."
    )
    parser.add_argument("--in_place",
                        "-i",
                        action="store_true",
                        help="do the operation in-place")
    args = parser.parse_args()
    main(**vars(args))
