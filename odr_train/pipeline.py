
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def get_pipeline(model):

    pipeline = Pipeline(steps=[
                # ('features', features_encoder),
                ('model', model)])

    return pipeline
