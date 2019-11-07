import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

class FFT:
    '''Holds a collection of Fast Fourier Transform implementations for 1D and 2D input'''

    def __init__(self):
        pass

    def fft(self, a, inverse=False):
        '''1-dimensional Fast Fourier Transform on a array'''
        n = len(a)
        if n == 1:
            return a
        a0 = [0j] * (n // 2)
        a1 = [0j] * (n // 2)
        for i in range(n // 2):
            a0[i] = a[i * 2]
            a1[i] = a[i * 2 + 1]
        y0 = self.fft(a0, inverse=inverse)
        y1 = self.fft(a1, inverse=inverse)
        angle = -1j * 2. * matplotlib.numpy.pi / n
        if inverse:
            angle *= -1.0
        w = matplotlib.numpy.exp(angle)
        w_base = 1 + 0j
        n2 = n // 2
        y = [0j] * n
        for i in range(n // 2):
            y[i] = y0[i] + w_base * y1[i]
            y[n2 + i] = y0[i] - w_base * y1[i]
            if inverse:
                y[i] /= 2.
                y[n2 + i] /= 2.
            w_base *= w
        return y


    def fft2(self, a, inverse=False):
        '''2-dimensional Fast Fourier Transform implementation. Generalized from 1-dimensional FFT'''

        n, m = a.shape
        arr = a * complex(1)
        for i in range(n):
            res = self.fft(arr[i], inverse=inverse)
            for k in range(m):
                arr[i][k] = res[k]
        # print(arr)
        arr = matplotlib.numpy.transpose(arr)
        for i in range(m):
            res = self.fft(arr[i], inverse=inverse)
            for k in range(n):
                arr[i][k] = res[k]
        return matplotlib.numpy.transpose(arr)

class Compressor:
    '''Class for compressing the images'''

    def __init__(self, directory='inputs', outputDirectory='AbdurasulRakhimovOutputs'):
        '''Initializer. Initializes the input directory and the output directory. '''

        self.fft = FFT()
        self.directory = directory
        self.outputDirectory = outputDirectory

    def compress(self, inputFilePath, outputFilePath):
        ''' Compresses the image in the location pointed by inputFilePath and
            writes the compressed image to outputFilePath'''

        image = plt.imread(inputFilePath)

        if len(image.shape) > 2:
            image = self.rgb2gray(image)
        
        image = self.fft.fft2(image)

        n = image.shape[0]
        pos = int(n * n * 95.0 / 100.0)
        image_flat = (matplotlib.numpy.absolute(image)).flatten()
        image_flat.sort()
        threshold = image_flat[pos]

        image = image * complex(1)
        for i in range(n):
            for j in range(n):
                image[i][j] = image[i][j] if matplotlib.numpy.absolute(image[i][j]) > threshold else 0.0
        
        image = self.fft.fft2(image, inverse=True)
        image = matplotlib.numpy.log(matplotlib.numpy.absolute(image))
        pil_image = Image.fromarray(image, mode='L')
        pil_image.save(outputFilePath)
        # image *= (255.0 / image.max())
        # image = matplotlib.numpy.ndarray.astype(image, dtype=matplotlib.numpy.uint8)

        # plt.imsave(outputFilePath, image, cmap='gray', format='TIFF')


    def compressAll(self):
        ''' Compresses all the pictures with extension .tif and writes the output
            to the specified folder in class initialization'''

        if not matplotlib.os.path.isdir(self.directory):
            print('Input directory does not exist')
            return
        
        if matplotlib.os.path.isfile(self.outputDirectory):
            print(f'File {self.outputDirectory} exists. Removing it and recreating a folder')
            matplotlib.os.remove(self.outputDirectory)
        
        if matplotlib.os.path.isdir(self.outputDirectory):
            print(f'Folder {self.outputDirectory} already exists. Removing it...')
            matplotlib.os.rmdir(self.outputDirectory)
        
        matplotlib.os.mkdir(self.outputDirectory)

        files = [f for f in matplotlib.os.listdir(self.directory)]

        for file in files:
            name, extension = matplotlib.os.path.splitext(file)
            if extension != '.tif':
                continue
            outputFilePath = self.outputDirectory + '/' + name + 'Compressed.tif'
            inputFilePath = self.directory + '/' + file
            self.compress(inputFilePath, outputFilePath)

    def rgb2gray(self, rgb):
        ''' converts rgb image to grayscale. Input is a numpy array with three dimensions:
            [N, M, 3]'''

        return matplotlib.numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

if __name__ == "__main__":
    
    compressor = Compressor()
    compressor.compressAll()