import logging

from plantpathology.utils import *
from plantpathology.models import *

import pickle

datasets = ["train", "test"]

models = [
    "BaselineCNN",
    "SENetTrial",
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
    logging.getLogger("matplotlib").setLevel(level=logging.ERROR)

    logging.basicConfig(level=logging.DEBUG)
    vanilla_df, vanilla_test_output_df = list(map(lambda basename: pd.read_csv("%s.csv" % basename), datasets))
    df, test_output_df = list(map(add_ext, (vanilla_df, vanilla_test_output_df)))

    validation_split = 0.3
    testing_split = 0.2
    training_split = 1 - validation_split - testing_split

    trainval_df, test_df = train_test_split(df, test_size = testing_split, random_state = 5489001)
    train_df, validation_df = train_test_split(trainval_df, test_size = validation_split * (1 - testing_split), random_state = 5489002)

    losses = {}

    metrics = ["loss", "roc_auc"]

    tee_val = lambda metric: (metric, "val_" + metric)

    def handle_history(model, history, test_metric, test_metric_value):
        for metric, val_metric in map(tee_val, metrics):
            fig, ax = plt.subplots(1)

            ax.plot(history.history[metric], label = "Train")
            ax.plot(history.history[val_metric], label = "Validation")
            ax.set_title("Model: %s - %s" % (model.name, metric))
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)

            fig.legend(loc = "upper right")
            fig.savefig("outputs/%(model_name)s_%(metric)s_%(test_metric)s_%(value).4f.png" % dict(model_name = model.name, metric = metric, test_metric = test_metric, value = test_metric_value))
            
            plt.close(fig)

    for model in map(get_model, models):
        # Training
        history = model.fit_df(train_df, validation_df)

        # NOTE: threeway splitting the training set
        test_metrics = model.evaluate_df(test_df)

        test_loss, test_auc_2 = test_metrics

        logger.info("model \"%s\": testing loss: %f", model.name, test_loss)

        losses[model.name] = dict(zip(metrics, test_metrics)) # { key: value.copy() for (key, value) in test_loss.items() }

        

        handle_history(model, history, "auc", test_auc_2)


        try:
            predicted = model.predict_df(test_output_df)
            data = {
                "image_id": vanilla_test_output_df.image_id
            }
            logger.info("model: %s, column_name: image_id, shape: %d", model.name, len(test_output_df.image_id))
            for column_name, values in zip(train_df.columns[1:], predicted.T):
                data[column_name] = values
                logger.info("model: %s, column_name: %s, shape: %r", model.name, column_name, values.shape)


            predicted_df = pd.DataFrame(data = data)
            predicted_df.to_csv("outputs/%s_predicted_auc_%.4f.csv" % (model.name, test_auc_2))
            model.save_weights("outputs/%s_predicted_auc_%.4f.h5" % (model.name, test_auc_2))
        except Exception as e:
            logger.exception(e)




    with open("losses.pickle", "wb") as f:
        pickle.dump(losses, f)

        


    



