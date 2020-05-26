import abc

class generic_profile(object):
    '''A generic base class to standardize the methods for the
    time profiles. While I'm only currently using scipy-based
    probability distributions, you can write your own if you
    want. Just be sure to define these methods and ensure that
    the PDF is normalized!
    '''

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,): pass

    @abc.abstractmethod
    def pdf(self, times): pass

    @abc.abstractmethod
    def logpdf(self, times): pass

    @abc.abstractmethod
    def random(self, n): pass

    @abc.abstractmethod
    def effective_exposure(self, times): pass

    @abc.abstractmethod
    def get_range(self): pass
