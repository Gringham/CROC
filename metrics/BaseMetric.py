import abc
import tqdm

class BaseMetric(metaclass=abc.ABCMeta):
    '''
    This class is an abstract class for metrics
    '''

    def __init__(self):
        pass
        #self.transform = transforms.ToTensor()

    #def img2tensor(self, np_image):
    #    '''
    #    This function converts an image to a tensor
    #    :param path: path to the image
    #    :return: tensor
    #    '''
    #    if type(np_image) == np.ndarray or np.array(np_image[0]).ndim < 3:
    #        if not type(np_image) == np.ndarray:
    #            np_image = np.array(np_image)
    #        return self.transform(np_image) 
    #    elif type(np_image) == list:
    #        return [self.transform(np.array(i)) for i in tqdm.tqdm(np_image, desc="Transforming images to tensors")]
    #        

    @abc.abstractmethod
    def __call__(self, gt, hyp):
        '''
        This function calculates a metric given all of its parameters
        :return: score
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self):
        '''
        This method returns an object holding information about the current configuration
        :return: ConfigurationVersionString
        '''
        raise NotImplementedError
