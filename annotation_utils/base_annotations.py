class BaseAnnotations:

    def __init__(self):
        pass

    @staticmethod
    def default_path() -> str:
        raise NotImplementedError("Derived classes must override the 'default_path' method")

    def convert(self):
        """Converts from the raw data to the classes specified format"""
        raise NotImplementedError("Derived classes must override the 'convert' method")

    def load(self, data_path: str):
        """Converts annotation file of specified format to universal data format"""
        raise NotImplementedError("Derived classes must override the 'load' method")

    def write(self, output_path: str, overwrite: bool = False):
        """Writes specified format annotations to disk"""
        raise NotImplementedError("Derived classes must override the 'write' method")

    def display(self):
        """Display annotations in the specified format"""
        raise NotImplementedError("Derived classes must override the 'draw' method")
