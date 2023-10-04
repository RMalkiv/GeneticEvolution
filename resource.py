# class ResourceType:
#     def __init__(self, resource_dict={}):
#         self.res = resource_dict
#
#     def __add__(self, other):
#         if isinstance(other, self.__class__):
#             keys = list(set(list(self.res.keys()) + list(other.res.keys())))
#             new_dict = {}
#             for key in keys:
#                 new_dict[key] = self.res.get(key, 0) + other.res.get(key, 0)
#             return ResourceType(resource_dict=new_dict)
#
#     def __sub__(self, other):
#         if isinstance(other, self.__class__):
#             keys = list(set(list(self.res.keys()) + list(other.res.keys())))
#             new_dict = {}
#             for key in keys:
#                 value = self.res.get(key, 0) - other.res.get(key, 0)
#                 if value != 0:
#                     new_dict[key] = value
#             return ResourceType(resource_dict=new_dict)
#
#     def __lt__(self, other):
#         if isinstance(other, self.__class__):
#             keys = list(set(list(self.res.keys()) + list(other.res.keys())))
#             for key in keys:
#                 self_value = self.res.get(key, 0)
#                 other_value = other.res.get(key, 0)
#                 if self_value >= other_value:
#                     return False
#             return True
#
#         elif isinstance(other, int):
#             keys = list(set(list(self.res.keys())))
#             for key in keys:
#                 self_value = self.res.get(key, 0)
#                 other_value = other
#                 if self_value >= other_value:
#                     return False
#             return True
#
#     def __gt__(self, other):
#         if isinstance(other, self.__class__):
#             keys = list(set(list(self.res.keys()) + list(other.res.keys())))
#             for key in keys:
#                 self_value = self.res.get(key, 0)
#                 other_value = other.res.get(key, 0)
#                 if self_value <= other_value:
#                     return False
#             return True
#
#         elif isinstance(other, int):
#             keys = list(set(list(self.res.keys())))
#             for key in keys:
#                 self_value = self.res.get(key, 0)
#                 other_value = other
#                 if self_value <= other_value:
#                     return False
#             return True
#
#     def __eq__(self, other):
#         if isinstance(other, self.__class__):
#             keys = list(set(list(self.res.keys()) + list(other.res.keys())))
#             for key in keys:
#                 self_value = self.res.get(key, 0)
#                 other_value = other.res.get(key, 0)
#                 if self_value != other_value:
#                     return False
#             return True
#
#         elif isinstance(other, int):
#             keys = list(set(list(self.res.keys())))
#             for key in keys:
#                 self_value = self.res.get(key, 0)
#                 other_value = other
#                 if self_value != other_value:
#                     return False
#             return True
#
#     def __floordiv__(self, other):
#         if isinstance(other, self.__class__):
#             keys = list(set(list(other.res.keys())))
#             new_dict = {}
#             for key in keys:
#                 new_dict[key] = self.res.get(key, 0) // other.res.get(key, 1)
#             return ResourceType(resource_dict=new_dict)
#
#         elif isinstance(other, int):
#             keys = list(set(list(self.res.keys())))
#             new_dict = {}
#             for key in keys:
#                 new_dict[key] = self.res.get(key, 0) // other
#             return ResourceType(resource_dict=new_dict)
#
#     def __mul__(self, other):
#         if isinstance(other, int):
#             keys = list(set(list(self.res.keys())))
#             new_dict = {}
#             for key in keys:
#                 new_dict[key] = self.res.get(key, 0) * other
#             return ResourceType(resource_dict=new_dict)
#         elif isinstance(other, float):
#             keys = list(set(list(self.res.keys())))
#             new_dict = {}
#             for key in keys:
#                 new_dict[key] = int(self.res.get(key, 0) * other)
#             return ResourceType(resource_dict=new_dict)
#
#     def min(self):
#         keys = self.res.keys()
#         minimum = min([self.res[key] for key in keys])
#         return minimum
#
#     def __repr__(self):
#         return str(self.res)


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