from utils.model import  Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

And = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}
df = pd.DataFrame(And)
df

x,y = prepare_data(df)

Eta = 0.3
epochs = 10

model = Perceptron(eta=Eta,epochs=epochs)
model.fit(x,y)
print("33333333")
model.total_loss()