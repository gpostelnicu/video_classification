import argparse

from video_classification.trainer import Trainer


def main(base_dir, train_list_file, test_list_file, config_file, num_frames, save_prefix,
         num_epochs, num_workers, print_every_n):
    trainer = Trainer(base_dir=base_dir, train_list_file=train_list_file, test_list_file=test_list_file,
                      config_file=config_file, num_frames=num_frames)
    trainer.train(save_prefix=save_prefix, num_epochs=num_epochs, num_workers=num_workers,
                  print_every_n=print_every_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--train_list', required=True)
    parser.add_argument('--test_list', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--num_frames', type=int, default=29,
                        help='Number of frames per video clip to consider.')
    parser.add_argument('--save_prefix', required=True)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--print_every_n', type=int, default=10)

    args = parser.parse_args()

    main(args.base_dir, args.train_list, args.test_list, args.config, args.num_frames,
         args.save_prefix, args.num_epochs, args.num_workers, args.print_every_n)
