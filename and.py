from utils.all_utils import save_Model, save_plot
from utils.model import  Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import os
import  logging
logging_str = "[%(asctime)s:%(levelname)s: %(module)s] %(message)s "
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,'ruuning_log.log'), level=logging.INFO,
                    format=logging_str,filemode='a')

def main(data,eta,epochs,modelfilename,plotpng):
    
    df = pd.DataFrame(data)
    logging.info(f"This is me{df}")

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
    try:
        logging.info("\n>>>> Starting training >>>>")
        main(data=And,eta=Eta,epochs=epochs,modelfilename='and.model',plotpng="and.png")
        logging.info(">>>> traning done >>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e