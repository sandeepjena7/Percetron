import  numpy as np

class Perceptron:
  def __init__(self,eta,epochs):
    self.weights = np.random.randn(3)*1e-4
    print(f"intial weight before training:\n {self.weights}")
    self.eta = eta # eta learning rate
    self.epochs = epochs

  def activationfunction(self,input,weights):
    z = np.dot(input,weights)
    return np.where(z>0,1,0)
  
  def fit(self,x,y):
    self.x = x
    self.y = y
    Weights_all = []


    x_with_bias = np.c_[self.x,-np.ones((len(self.x),1))]
    print(f"X with bias:\n {x_with_bias}")

    for epoch in range(self.epochs):
      print(f"for epoch:\t {epoch}")
      print("--"*10)
      Weights_all.append(self.weights)
      y_hat = self.activationfunction(x_with_bias,self.weights)# foraward propagation
      print(f"Predicted value after forward pass:\n{y_hat}")

      self.error = self.y-y_hat
      # print(f"error:\n{self.error}")
      print(f"Total loss is {np.sum(self.error)}")
      if np.sum(self.error) ==0:
        
        break


      self.weights = self.weights + self.eta*np.dot(x_with_bias.T,self.error)# backward propagation
      print(f"updated weights after epoch:\n {epoch}/{self.epochs} :\n {self.weights} ")
      print("===="*30)
    # print(Weights_all)

  
  def predict(self,x):
    x_with_bias = np.c_[x,-np.ones((len(x),1))]
    return self.activationfunction(x_with_bias,self.weights)
  def total_loss(self):
    toatal_loss = np.sum(self.error)
    print(f"total loss: {toatal_loss}")
    return toatal_loss