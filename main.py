import wandb
import tensorflow as tf


from model import Model
from dataset import SentDataset

def main(self):

    dataset = SentDataset('imdb')
    wandb.init(project = 'sturectured_selfattentive')
    params = {'vocab_size': dataset.vocab_size}

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    configs = tf.contrib.learn.RunConfig(save_summary_steps = 1000,
                                        keep_checkpoint_max = 2,
                                        session_config = session_config)
    model = Model()
    estimator = tf.estimator.Estimator(model_fn = model.model_function,
                                        params = params,
                                        model_dir = './check_point',
                                        config = configs)
    print('Training start')
    estimator.train(dataset.train_input_function, hooks=[wandb.tensorflow.WandbHook(steps_per_log=100)])
    estimator.evaluate(dataset.test_input_function)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
