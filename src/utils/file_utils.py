import os
import json
import time


def write_file(dir_path, file_name, content) -> str:
    """
    Write content to a file.

    :param file_path: Path to the file.
    :param content: Content to write to the file.
    """

    # if folder does not exist, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # if file already exists, delete it
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Write the content to the file
    with open(file_path, "w") as file:
        file.write(content)
    print(f"Content written to {file_path}")

    return file_path


def read_jsonl_file(file_path: str) -> list:
    """
    Read a JSONL file and return a list of dictionaries.

    :param file_path: Path to the JSONL file.
    :return: List of dictionaries.
    """
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def read_json_file(file_path: str) -> dict:
    """
    Read a JSON file and return its content.

    :param file_path: Path to the JSON file.
    :return: Content of the JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def get_unique_name_for_file_name(file_name: str) -> str:
    """
    Get a unique file name by appending the current timestamp to the file name.

    :param file_name: Original file name.
    :return: Unique file name.
    """
    # Get the current timestamp
    timestamp = int(time.time())
    # Append the timestamp to the file name
    unique_file_name = f"{timestamp}_{file_name}"
    return unique_file_name
