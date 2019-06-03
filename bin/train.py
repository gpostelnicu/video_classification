import argparse

from video_classification.trainer import Trainer


def main(base_dir, train_list_file, test_list_file, config_file, num_epochs):
    trainer = Trainer(base_dir=base_dir, train_list_file=train_list_file, test_list_file=test_list_file,
                      config_file=config_file)
    trainer.train(num_epochs=num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--train_list', required=True)
    parser.add_argument('--test_list', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--num_epochs', type=int, default=1)

    args = parser.parse_args()

    main(args.base_dir, args.train_list, args.test_list, args.config, args.num_epochs)
