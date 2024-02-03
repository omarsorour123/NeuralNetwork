import numpy as np
class Loss:
    def calculate(self,output,y):
        initial_loss = self.forward(output,y)
        loss = np.mean(initial_loss)
        return loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_true,y_pred):
        samples = len(y_pred) # how many batches and rows
        y_pred_cliped = np.clip(y_pred,1e-7,1-1e-7) #prevent from loss = 0 and then loss = inf

        if len(y_true.shape) == 1: #SparseCategoricalCrossEntropy
            #loop over each row and get the y_true matching index
            correct_confidences = y_pred_cliped[range(samples),y_true]
        elif len(y_true.shape)== 2:
            correct_confidences = np.sum((y_pred_cliped*y_true),axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

class MeanSquaredError(Loss):
    def forward(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))