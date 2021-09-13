from utils.all_utils import save_Model, save_plot
from utils.model import  Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(data,eta,epochs,modelfilename,plotpng):
    
    df = pd.DataFrame(data)
    df

    x,y = prepare_data(df)

    

    model = Perceptron(eta=eta,epochs=epochs)
    model.fit(x,y)

    model.total_loss()

    save_Model(model,filename=modelfilename)
    save_plot(df,plotpng,model)

if __name__ == '__main__':
    And = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,0,0,1]
    }
    Eta = 0.3
    epochs = 10
    main(data=And,eta=Eta,epochs=epochs,modelfilename='and.model',plotpng="and.png")