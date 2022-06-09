import matplotlib.pyplot as plt
import torch
from sklearn import metrics
import numpy as np
from sklearn.metrics import plot_confusion_matrix
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_loss = np.load('losses_train.npy')
test_loss = np.load('losses_test.npy')
print("Min test loss:", min(test_loss))
print("Min train loss:", min(train_loss))
accuracy_train = np.load('accuracies_train.npy')
accuracy_test = np.load('accuracies_validations.npy')
print("Max test accuracy:", max(accuracy_test))
print("Max train accuracy:", max(accuracy_train))

# Train Test loss graph
plt.plot(train_loss, label='train loss', color='hotpink')
plt.plot(test_loss, label='test loss', color='blue')
plt.legend()
plt.xlabel('ephoc')
plt.ylabel('loss')
plt.title('loss as function of ephocs')
plt.show()

# Train Test accuracy graph
plt.plot(accuracy_train, label='train accuracy', color='hotpink')
plt.plot(accuracy_test, label='test accuracy', color='blue')
plt.legend()
plt.xlabel('ephoc')
plt.ylabel('accuracy')
plt.title('accuracy as function of ephocs')
plt.show()

confusion_train = np.array(np.load('confusion_mat_train.npy'))
confusion_test = np.array(np.load('confusion_mat_test.npy'))

# confusion matrix train-
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_train, display_labels=['neutral', 'happiness', 'sadness'])
disp.plot()
plt.title("confusion matrix- train")
plt.show()


# confusion matrix test-
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_test, display_labels=['neutral', 'happiness', 'sadness'])
disp.plot()
plt.title("confusion matrix- test")
plt.show()
