import pickle

def save_dataset(file_path, data):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        return True
    except Exception as e:
        return False

def load_dataset(file_path):
    try:
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
    except Exception as e:
        return None

def append_to_dataset(file_path, data_to_append):
    try:
        loaded_data = load_dataset(file_path)
        if loaded_data is not None:
            if isinstance(loaded_data, list):
                if isinstance(data_to_append, list):
                    loaded_data.extend(data_to_append)
                    with open(file_path, 'wb') as file:
                        pickle.dump(loaded_data, file)
                    return True
            else:
                return False
    except Exception as e:
        return False
