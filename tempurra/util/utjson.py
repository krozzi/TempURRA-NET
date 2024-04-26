import json
import os

def append_json_exit(new_data, dictn, filename='data.json'):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data[dictn].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

def append_json_exit_raw(new_data, dictn, filename='data.json'):

    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data[dictn] = new_data
        file.seek(0)
        json.dump(file_data, file, indent=4)

def append_json(json_data, filename='data.json'):
    with open(filename, "r+") as file:
        data = json.load(file)
        data.update(json_data)
        file.seek(0)
        json.dump(data, file, indent=4)


def edit_json(selector, key, file_name, target, value):
    try:
        # Open the JSON file and load the data
        with open(file_name, 'r') as file:
            data = json.load(file)

        # Find the user data to be edited
        user_data = next((user for user in data['user_data'] if user[selector] == key), None)

        user_data[target] = value

        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4)

    except FileNotFoundError:
        print("The 'user_data.json' file does not exist.")
    except json.JSONDecodeError:
        print("The 'user_data.json' file is not a valid JSON file.")


def create_json(filename):
    with open(filename, 'w') as file:
        file.write('{\n     "user_data" : []\n}')
        file.close()
    return file

def get_key_value(json, key):
    return json.get(key)

def read_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def read_json_raw(filename):
    with open(filename, 'w') as file:
        return file