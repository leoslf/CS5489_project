import logging

from plantpathology.utils import *
from plantpathology.models import *

import pickle

datasets = ["train", "test"]

models = [
    "BaselineCNN",
]

def get_model(name, *argv, **kwargs):
    return globals()[name](*argv, **kwargs)

def add_ext(df):
    df["image_id"] += ".png" # ".jpg"
    return df # new_df

if __name__ == "__main__":
    # Muting PIL
    logging.getLogger("PIL").setLevel(level=logging.ERROR)
    # Muting Tensorflow warning
    logging.getLogger("tensorflow").setLevel(level=logging.ERROR)

    logging.basicConfig(level=logging.DEBUG)
    vanilla_df, vanilla_test_output_df = list(map(lambda basename: pd.read_csv("%s.csv" % basename), datasets))
    df, test_output_df = list(map(add_ext, (vanilla_df, vanilla_test_output_df)))

    validation_split = 0.3
    testing_split = 0.2
    training_split = 1 - validation_split - testing_split

    trainval_df, test_df = train_test_split(df, test_size = testing_split, random_state = 5489001)
    train_df, validation_df = train_test_split(trainval_df, test_size = validation_split * (1 - testing_split), random_state = 5489002)

    losses = {}

    for model in map(get_model, models):
        # Training
        history = model.fit_df(train_df, validation_df)

        # NOTE: threeway splitting the training set
        test_loss = model.evaluate_df(test_df)

        logger.info("model \"%s\": testing loss: %f", model.name, test_loss)

        losses[model.name] = { key: value.copy() for (key, value) in test_loss.items() }

        predicted = model.predict_df(test_df)
        with open("%s_predicted_auc_%.4f.pickle" % (model.name, test_loss["auc_2"]), "wb") as f:
            pickle.dump(predicted, f)

        predicted_df = pd.DataFrame(data = dict(zip(train_df.columns, [vanilla_test_output_df.image_id] + predicted.T.tolist())))
        predicted_df.to_csv("%s_predicted_auc_%.4f.csv" % (model.name, test_loss["auc_2"]))




    with open("losses.pickle", "wb") as f:
        pickle.dump(losses, f)

        


    



