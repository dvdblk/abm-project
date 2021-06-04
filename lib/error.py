

class MinorityGameError(Exception):
    """Error class for any errors related to MinorityGame"""
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message
