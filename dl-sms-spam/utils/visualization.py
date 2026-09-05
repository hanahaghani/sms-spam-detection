#import library
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#confusion-matrix
def plot_confusion_matrix(y_true,y_preb,save_path):

    cm=confusion_matrix(y_true,y_preb)
    plt.imshow(cm)
    plt.savefig(save_path)
    plt.show()

#training loss vs validaition loss
def plot_training_history(training_losses,valid_losses,save_path):

    plt.figure(figsize=(10,8))
    epochs=range(1,len(training_losses)+1)
    plt.plot(epochs,training_losses,label="training loss")
    plt.plot(epochs,valid_losses,label='validation loss')
    plt.xlabel=('Epoch')
    plt.ylabel=('loss')
    plt.title("training and validation loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.close()

#neural network architecture
def plot_neural_network(model,save_path):
    pass
