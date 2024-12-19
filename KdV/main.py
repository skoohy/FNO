import wandb
from ml_collections import config_flags
from absl import app, flags

import train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config")


def main(argv):
    use_wandb = FLAGS.config.use_wandb

    if use_wandb == True:
        wandb.init(
            project=FLAGS.config.wandb.project,
            name=FLAGS.config.wandb.name,
            tags=FLAGS.config.wandb.tags,
            group=FLAGS.config.wandb.group,
        )
        run = wandb.run
        model_state = train.train_model(FLAGS.config, run)
        wandb.finish()
    else:
        run = None
        train.train_model(FLAGS.config, run)
    return model_state, run


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
