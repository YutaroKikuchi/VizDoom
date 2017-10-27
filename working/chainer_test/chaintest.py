import chainer
import matplotlib
import matplotlib.pyplot as plt
 
 
# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist(ndim=1)
 
ROW = 4
COLUMN = 5
for i in range(ROW * COLUMN):
    # train[i][0] is i-th image data with size 28x28
    image = train[i][0].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, i+1)          # subplot with size (width 3, height 5)
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('label = {}'.format(train[i][1]))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('images/mnist_plot.png')
plt.show()