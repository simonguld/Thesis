import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--delete_if_successful', type=str, default=0,)
    parser.add_argument('--test_mode', type=int, default=0,)
    args = parser.parse_args()

    test_mode = bool(args.test_mode)
    delete_if_successful = bool(args.delete_if_successful)

    print(test_mode)
    print(delete_if_successful)



if __name__ == '__main__':
    main()