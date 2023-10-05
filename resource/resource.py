class ResourceType:
    def __init__(self, resource_dict=None):
        self.res = resource_dict if resource_dict is not None else {}

    def _keys(self, other=None):
        if other is not None and isinstance(other, self.__class__):
            return set(self.res.keys()) | set(other.res.keys())
        return set(self.res.keys())

    def _get_value(self, key, other=None):
        if other is not None and isinstance(other, self.__class__):
            return self.res.get(key, 0), other.res.get(key, 0)
        return self.res.get(key, 0), 0

    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_dict = {key: sum(self._get_value(key, other)) for key in self._keys(other)}
            return ResourceType(new_dict)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            new_dict = {key: self._get_value(key, other)[0] - self._get_value(key, other)[1]
                        for key in self._keys(other)}
            return ResourceType(new_dict)

    def __lt__(self, other):
        for key in self._keys(other):
            if self._get_value(key, other)[0] >= self._get_value(key, other)[1]:
                return False
        return True

    def __gt__(self, other):
        for key in self._keys(other):
            if self._get_value(key, other)[0] <= self._get_value(key, other)[1]:
                return False
        return True

    def __eq__(self, other):
        for key in self._keys(other):
            if self._get_value(key, other)[0] != self._get_value(key, other)[1]:
                return False
        return True

    def __floordiv__(self, other):
        if isinstance(other, self.__class__):
            new_dict = {key: self._get_value(key, other)[0] // max(1, self._get_value(key, other)[1])
                        for key in self._keys(other)}
            return ResourceType(new_dict)
        elif isinstance(other, int):
            new_dict = {key: value // max(1, other) for key, value in self.res.items()}
            return ResourceType(new_dict)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_dict = {key: int(value * other) for key, value in self.res.items()}
            return ResourceType(new_dict)

    def min(self):
        return min(self.res.values())

    def scarcity_check(self, threshold):
        """Return keys of resources that are below a specified threshold."""
        return [key for key, value in self.res.items() if value < threshold]

    def normalize(self, new_sum=100):
        """Scale the resources so that they add up to `new_sum`, maintaining their proportion."""
        total = sum(self.res.values())
        if total == 0:  # avoid division by zero
            return self
        scale_factor = new_sum / total
        new_dict = {key: value * scale_factor for key, value in self.res.items()}
        return ResourceType(new_dict)

    def __repr__(self):
        return str(self.res)