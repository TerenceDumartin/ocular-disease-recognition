from odr_train.data import get_data, save_model_to_gcp, save_image_to_gcp
from odr_train.model import get_model_binary, get_model_classifier, get_model_classifier_vgg16, get_model_vgg16
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
        
        # infos
        self.name = kwargs.get("name", 'unnamed')
        
        #Model Params
        self.mode = kwargs.get("mode", "v0")
        self.target_v0 = kwargs.get("target_v0", 'N')
        self.target_v1 = kwargs.get("target_v1", ['D','G','C','A','H','M','O'])
        self.epochs = kwargs.get("epochs", 10)
        self.vgg16 = kwargs.get("vgg16", False)
        
        if self.mode == 'v1':
            self.tarstr = ''.join(self.target_v1)
        else:
            self.tarstr = self.target_v0
            
        
        #Options
        self.local = kwargs.get("local", True)
        self.save_model = kwargs.get("save_model", False)
        self.resize = kwargs.get("resize", False)
        self.mlflow = kwargs.get("mlflow", False)

    def retrieve_data(self):

        # get data
        self.y_train, self.y_test, self.X_train, self.X_test = \
            get_data(self.local, self.mode, self.target_v0, self.target_v1)

    def mlflow_log_run(self):

        # create a mlflow training
        self.mlflow_create_run()

        # log params
        self.mlflow_log_param("model_name", self.name)
        self.mlflow_log_param("epochs", self.epochs)

        # push metrics to mlflow
        self.mlflow_log_metric(self.score_name, self.score)

    def evaluate_model(self):

        # make prediction for metrics
        
        self.score_name =list(self.history.history.keys())[1]
        self.val_score_name =list(self.history.history.keys())[-1]
        self.score = round(self.history.history[self.val_score_name][-1]*100,2)
        
        print(f'{self.score_name} = {self.score}')

    def save_fig(self):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['loss'], label='train')
        ax1.plot(self.history.history['val_loss'], label='val')
        # ax1.set_ylim(0., 2.2)
        ax1.set_title('loss')
        ax1.legend()

        ax2.plot(self.history.history[self.score_name], label=f'train {self.score_name}' )
        ax2.plot(self.history.history[self.val_score_name], label=f'val {self.score_name}' )
        # ax2.set_ylim(0.25, 1.)
        ax2.set_title(self.score_name)
        ax2.legend()

        fig_name= self.name + '_loss_score_plot.png'
        fig_path = DATA_FOLDER + fig_name
        print(f'Saving loss/acc curves at {fig_path}')
        f.savefig(fig_path)

        if not self.local:
            print(f'Exporting figure to GC storage')
            save_image_to_gcp(fig_name)
    

    def fit_model(self):
        es = EarlyStopping(patience=5, restore_best_weights=True)

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
        in_shape = self.X_train[0].shape
        if self.mode == 'v0':
            if self.vgg16:
                self.model = get_model_vgg16(in_shape)
            else:
                self.model = get_model_binary()
        else:
            out_shape = len(self.y_train.columns)
            if self.vgg16:
                self.model = get_model_classifier_vgg16(in_shape, out_shape)
            else:
                self.model = get_model_classifier(out_shape)

        # step 3 : train
        self.fit_model()

        # step 4 : evaluate perf
        self.evaluate_model()
        
        #---rename our run

        self.name = f"{self.name}_{self.mode}_{self.tarstr}_{int(self.score)}"
        
        # step 5 : save training loss score
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
                #Basics infos
                name            =  "baseline_vgg",

                #Model Params
                mode            = 'v0', #('v0' = predict N or C on all df , 'v1' = desease classifier on desease df)
                target_v0       = 'N',   # choosing y (if mode = all) ex : 'C'
                epochs          = 2,
                vgg16           = True, #Transfert learning
                
                #Options
                local           = True,  #for taking data in local or in gcp
                save_model      = False,  #for saving the model
                resize          = False,  #add resizing in pipeline (useless for now)
                mlflow          = False,  #export results in MLFlow
                
            ),
            dict(
                #Basics infos
                name            =  "baseline_vgg",

                #Model Params
                mode            = 'v1', #('v0' = predict N or C on all df , 'v1' = desease classifier on desease df)
                target_v0       = 'N',   # choosing y (if mode = all) ex : 'C'
                target_v1       = ['D','G','C','A','H','M','O'], #deseases to classify (if mode = deseases) ex :['D','G','C','A','H','M','O']
                epochs          = 2,
                vgg16           = True, #Transfert learning
                
                #Options
                local           = True,  #for taking data in local or in gcp
                save_model      = False,  #for saving the model
                resize          = False,  #add resizing in pipeline (useless for now)
                mlflow          = False,  #export results in MLFlow
                
            ),
            dict(
                #Basics infos
                name            =  "baseline_vgg",

                #Model Params
                mode            = 'v1', #('v0' = predict N or C on all df , 'v1' = desease classifier on desease df)
                target_v0       = 'N',   # choosing y (if mode = all) ex : 'C'
                target_v1       = ['G','C','A','H','M','O'], #deseases to classify (if mode = deseases) ex :['D','G','C','A','H','M','O']
                epochs          = 2,
                vgg16           = True, #Transfert learning
                
                #Options
                local           = True,  #for taking data in local or in gcp
                save_model      = False,  #for saving the model
                resize          = False,  #add resizing in pipeline (useless for now)
                mlflow          = False,  #export results in MLFlow
            ),

    ]
    
    for params in param_set : 

        trainer = Trainer(**params)
        trainer.train()
