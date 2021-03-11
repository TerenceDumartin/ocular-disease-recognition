
from odr_train.data import get_data, save_model_to_gcp, save_image_to_gcp
from odr_train.model import get_model_binary
from odr_train.pipeline import get_pipeline
from odr_train.mlf import MLFlowBase
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import joblib

DATA_FOLDER = "data/Train_test_split/"  #must be the same as data.py

class Trainer(MLFlowBase):

    def __init__(self, **kwargs):
        super().__init__(
            "[FR] [Bdx] [527] ODR",
            "https://mlflow.lewagon.co")
        
        self.local = kwargs.get("local", True)
        self.target = kwargs.get("target", 'C')
        self.resize = kwargs.get("resize", False)
        self.mlflow = kwargs.get("mlflow", False)
        self.epochs = kwargs.get("epochs", 50)
        self.name = kwargs.get("name", 'unnamed')
        self.save_model = kwargs.get("save_model", False)

    def retrieve_data(self):

        # get data
        df_train, df_test, self.X_train, self.X_test = get_data(self.local)
        
        self.y_train = df_train[self.target]
        self.y_test = df_test[self.target]

    def mlflow_log_run(self):

        # create a mlflow training
        self.mlflow_create_run()

        # log params
        self.mlflow_log_param("model_name", self.name)
        self.mlflow_log_param("epochs", self.epochs)

        # push metrics to mlflow
        self.mlflow_log_metric("accuracy", self.accuracy)

    def evaluate_model(self):

        # make prediction for metrics
        # DEPEND ON TF VERSION!!!!
        if 'val_accuracy' in self.history.history.keys():
            self.acc_name = 'accuracy'
            self.val_acc_name = 'val_accuracy'
            self.accuracy = round(self.history.history[self.val_acc_name][-1]*100,2)
        elif 'val_acc' in self.history.history.keys():
            self.acc_name = 'acc'
            self.val_acc_name = 'val_acc'
            self.accuracy = round(self.history.history[self.val_acc_name][-1]*100,2)
        else:
            self.accuracy = 0
            print('WARNING - Accuracy not found')
            
        print(f'accuracy = {self.accuracy}')

    def save_fig(self):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['loss'], label='train')
        ax1.plot(self.history.history['val_loss'], label='val')
        ax1.set_ylim(0., 2.2)
        ax1.set_title('loss')
        ax1.legend()

        ax2.plot(self.history.history[self.acc_name], label='train accuracy' )
        ax2.plot(self.history.history[self.val_acc_name], label='val accuracy' )
        ax2.set_ylim(0.25, 1.)
        ax2.set_title('Accuracy')
        ax2.legend()

        fig_name= self.name + '_loss_acc_plot.png'
        fig_path = DATA_FOLDER + '/' + fig_name
        print(f'Saving loss/acc curves at {fig_path}')
        f.savefig(fig_path)

        if not self.local:
            print(f'Exporting figure to GC storage')
            save_image_to_gcp(fig_name)
    

    def fit_model(self):
        es = EarlyStopping(patience=20, restore_best_weights=True)
        self.history = self.model.fit(self.X_train, self.y_train,
                                validation_data=(self.X_test, self.y_test),
                                epochs=self.epochs,
                                batch_size=16, 
                                verbose=1,
                                callbacks=[es])

    def savemodel(self):
        file_name = f'model_{self.name}.h5'
        if self.local:
            self.model.save(f"./data/models/{file_name}")
            # joblib.dump(self.model, f"data/models/{joblib_name}")
            print(f'model saved at data/models/{file_name}')
        else:
            self.model.save(file_name)
            # joblib.dump(self.model, joblib_name)
            save_model_to_gcp(file_name)
            
    def train(self):
        # step 1 : get data
        self.retrieve_data()

        # step 2 : create model
        self.model = get_model_binary()

        # step 3 : train
        self.fit_model()

        # step 4 : evaluate perf
        self.evaluate_model()
        
        #---rename our run
        self.name = self.name + '_' + self.target + '_' + self.accuracy
        
        # step 5 : save training loss accuracy
        self.save_fig()
        
        # step 6 : save the trained model
        if self.save_model :
            self.savemodel()
        
        # step 7 : log run in mlflow
        if self.mlflow:
            self.mlflow_log_run()

        print('Finito cappuccino!')


if __name__ == '__main__':
    param_set = [
            dict(
                local           = False,  #for taking data in local or in gcp
                epochs          = 100,
                save_model      = True, #for saving the model
                target          = 'C',   # choosing y 
                resize          = False, #add resizing in pipeline
                mlflow          = True, #export results in MLFlow
                name            =  "baseline"
            ),
            dict(
                local           = False,  #for taking data in local or in gcp
                epochs          = 100,
                save_model      = True, #for saving the model
                target          = 'N',   # choosing y 
                resize          = False, #add resizing in pipeline
                mlflow          = True, #export results in MLFlow
                name            =  "baseline"
            )
    ]
    
    for params in param_set : 

        trainer = Trainer(**params)
        trainer.train()
