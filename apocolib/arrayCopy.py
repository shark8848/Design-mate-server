class arrayCopy:

    @staticmethod
    def arrayCopyByRange(original_array, i, j):
        try:
            if isinstance(slice(i, j), slice):
                # Convert slice object to a valid range of indices
                indices = range(*slice(i, j).indices(len(original_array)))
                new_array = [original_array[k] for k in indices]
                return new_array
            else:
                raise TypeError("Invalid slice object")
        except Exception as e:
            raise arrayCopyByRangeException(f"arrayCopyByRangeException,Check if the array is empty or if there is a range exception.{str(e)}")

class arrayCopyByRangeException(Exception):
    def __init__(self, message):
        self.message = message
