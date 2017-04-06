import argparse
from mlp import util
from rnn import BPTT


def get_sets(S, _dir, _ext, name=None):
    X = []
    Y = []
    if name:
        print(name + " Sets:")
    for s in S:
        _file = _dir + s + _ext
        x, lbls = util.load_data_file(_file)
        print('--loaded:', _file)
        y = util.to_output_vector(lbls)
        X.append(x)
        Y.append(y)
    return X, Y


def parse_args(_args=None):
    parser = argparse.ArgumentParser(description='Run a multilayer perceptron via vector operations')
    parser.add_argument('--num_features', '-f', type=int, default=6, help='number of features')
    parser.add_argument('--num_classes', '-c', type=int, default=7, help='number of output classes')
    parser.add_argument('--num_hidden', '-H', type=int, default=60, help='number of hidden nodes to use')
    parser.add_argument('-dir', default="../experiments/data/pills/pills_0", help='if specified, prepends all the data files with this string')
    parser.add_argument('-ext', default=".txt", help='if specified, appends all the data files with this string')
    parser.add_argument('--train', '-Tr', nargs="+", default=["1", "2", "6", "7"], help='training set data files')
    parser.add_argument('--test', '-Ts', nargs="+", default=["3", "4", "5"], help='test set data files')
    parser.add_argument('--validation', '-V', nargs="*", default=["8", "9"], help='validation set data files')
    parser.add_argument('--learning_rate', '-l', type=float, default=.9, help='learning rate to use')
    parser.add_argument('--max_epochs', '-e', type=int, default=1000, help='maximum number of epochs to allow')
    parser.add_argument('--patience', '-p', type=int, default=20, help='patience param')
    parser.add_argument('--k_back', '-kb', type=int, default=1, help='k back param')
    parser.add_argument('--k_forward', '-kf', type=int, default=1, help='k forward param')
    parser.add_argument('--out', '-o', default="../experiments/results/bptt.txt",
                        help='The output directory to save results to')
    if _args is None:
        return parser.parse_args()
    return parser.parse_args(_args)


if __name__ == '__main__':
    args = parse_args()
    train = get_sets(args.train, args.dir, args.ext, "Training")
    test = get_sets(args.test, args.dir, args.ext, "Test")
    validation = get_sets(args.validation, args.dir, args.ext, "Validation")
    model = BPTT(args.num_features, args.num_hidden, args.num_classes,
                 validation_set=validation, multi_vsets=True,
                 k_back=args.k_back, k_forward=args.k_forward,
                 max_epochs=args.max_epochs, patience=args.patience, learning_rate=args.learning_rate)
    num_epochs = model.fit(train[0], train[1], True)
    score = model.score(test[0], test[1], multi_sets=True)
    print("accuracy:", score)
    print("epochs:", num_epochs)
    if args.out:
        with open(args.out, 'a') as _f:
            print("Training Sets:", args.train, "Validation Sets:", args.validation, "Test Sets:", args.test, file=_f)
            print("hidden nodes:", args.num_hidden, file=_f)
            print("learning rate:", args.learning_rate, "patience:", args.patience, "max epochs:", args.max_epochs, file=_f)
            print("accuracy:", score, file=_f)
            print("epochs:", num_epochs, file=_f)
            print("k:", args.k_back, args.k_forward, file=_f)
            print(file=_f)
