from abc import ABCMeta

class ABCRegistry(ABCMeta):
    """Extends the ABCMeta class to include a registry
    """
    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        if not hasattr(new_cls,'_registry_name'):
            raise Exception('Any class with ABCRegistry as metaclass has to\
             define the class attribute _registry_name')
        cls.REGISTRY[new_cls._registry_name] = new_cls
        return ABCMeta.__new__(cls,name,bases, attrs)


    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
