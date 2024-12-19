import train
import wandb
from ml_collections import config_flags
from absl import app, flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config")


def main(argv):
    config = FLAGS.config

    sweep_config = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "train_loss"},
    }

    # add hyper-parameters to sweep over here
    parameters_dict = {
        "modes": {"values": [4, 8]},
        "width": {"values": [16, 32]},
    }

    sweep_config["parameters"] = parameters_dict

    def train_sweep():
        config = FLAGS.config

        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.tags,
            group=config.wandb.group,
        )

        sweep_config = wandb.config

        wandb.run.name = "custom_name_here"
        wandb.run.save()

        config.arch.modes = sweep_config.modes

        train.train_model(config, wandb.run)

        wandb.finish()

    sweep_id = wandb.sweep(sweep_config, project=config.wandb.project)

    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
