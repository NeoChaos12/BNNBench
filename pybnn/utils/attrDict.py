class AttrDict(dict):
    r"""
    Accessing dictionary keys as attributes is *so* much more convenient.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self