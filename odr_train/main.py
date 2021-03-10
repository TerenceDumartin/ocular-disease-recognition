
from odr_train.data import get_data, save_model_to_gcp
from odr_train.model import get_model_binary
from odr_train.pipeline import get_pipeline
from odr_train.mlf import MLFlowBase
from tensorflow.keras.callbacks import EarlyStopping

import joblib

class Trainer(MLFlowBase):

    def __init__(self, **kwargs):
        super().__init__(
            "[FR] [Bdx] [527] ODR",
            "https://mlflow.lewagon.co")
        
        self.local = kwargs.get("local", True)
        self.target = kwargs.get("target", 'C')
        self.resize = kwargs.get("resize", False)
        self.mlflow = kwargs.get("mlflow", False)
        self.name = kwargs.get("name", False)
        self.epochs = kwargs.get("epochs", 50)
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
        self.accuracy = round(self.history.history['val_accuracy'][-1]*100,2)
        print(f'accuracy = {self.accuracy}')


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

        # # step 2 : create model
        # self.model = get_model_binary()

        # # step 3 : train
        # self.fit_model()

        # # step 4 : evaluate perf
        # self.evaluate_model()
        
        # # step 5 : save the trained model
        # if self.save_model :
        #     self.savemodel()
        
        # # step 7 : log run in mlflow
        # if self.mlflow:
        #     self.mlflow_log_run()

        print('Finito cappuccino!')


if __name__ == '__main__':
    param_set = [
            dict(
                local           = False,  #for taking data in local or in gcp
                epochs          = 1,
                save_model      = True, #for saving the model
                target          = 'C',   # choosing y 
                resize          = False, #add resizing in pipeline
                mlflow          = True, #export results in MLFlow
                name            =  "test_package2"
            )
    ]
    
    for params in param_set : 
        trainer = Trainer(**params)
        trainer.train()
