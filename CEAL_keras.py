import argparse
import os

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_contrib.applications.resnet import ResNet18


def initialization():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    n_classes = np.max(y_test) + 1

    # Convert class vectors to binary class matrices.
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    x_train /= 128.
    x_test /= 128.

    datagen = ImageDataGenerator()
    initial_train_size = int(x_train.shape[0] * args.initial_annotated_perc)

    x_train_initial, y_train_initial = iter(
        datagen.flow(x_train, y_train, batch_size=initial_train_size, shuffle=True)).next()

    if os.path.exists(args.chkt_filename):
        model = load_model(args.chkt_filename)
    else:
        model = ResNet18((x_train[-1,].shape), n_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        model.fit(x_train_initial, y_train_initial, validation_data=(x_test, y_test), batch_size=args.batch_size,
                  epochs=args.epochs, verbose=args.verbose, callbacks=[checkpoint])

    scores = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=args.verbose)
    print('Initial Test Loss: ', scores[0], ' Initial Test Accuracy: ', scores[1])

    return model, x_train, y_train, x_test, y_test, n_classes


# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index,
                           max_prob,
                           pred_label))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]


# TODO finish implementation
# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)
    msi = np.column_stack((origin_index,
                           margim_sampling,
                           pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    # entropy = stats.entropy(y_pred_prob.T)
    # entropy = np.nan_to_num(entropy)
    origin_index = np.arange(0, len(y_pred_prob))
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)
    eni = np.column_stack((origin_index,
                           entropy,
                           pred_label))

    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]


def get_high_confidence_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    hcs = eni[eni[:, 1] < delta]
    return hcs[:, 2].astype(int), hcs[:, 0].astype(int)


def get_uncertain_samples(y_pred_prob, n_samples, criteria='least_confidence'):
    if criteria == 'lc':
        return least_confidence(y_pred_prob, n_samples)
    elif criteria == 'ms':
        return margin_sampling(y_pred_prob, n_samples)
    elif criteria == 'en':
        return entropy(y_pred_prob, n_samples)
    else:
        raise ValueError(
            'Unknown criteria value \'%s\', use one of [\'lc\',\'ms\',\'en\']' % criteria)


def run_ceal(args):
    # TODO exclude samples from initial train
    model, x_train, y_train, x_test, y_test, n_classes = initialization()
    w, h, c = x_train[-1,].shape

    # unlabeled samples
    DU = x_train, y_train

    # initially labeled samples
    DL = np.empty((0, w, h, c)), np.empty((0, n_classes))

    # high confidence samples
    DH = np.empty((0, w, h, c)), np.empty((0, n_classes))

    for i in range(args.maximum_iterations):

        y_pred_prob = model.predict(DU[0], verbose=args.verbose)

        _, un_idx = get_uncertain_samples(y_pred_prob, args.uncertain_samples_size, criteria=args.uncertain_criteria)
        DL = np.append(DL[0], np.take(DU[0], un_idx, axis=0), axis=0), \
             np.append(DL[1], np.take(DU[1], un_idx, axis=0), axis=0)

        if args.cost_effective:
            hc_labels, hc_idx = get_high_confidence_samples(y_pred_prob, args.delta)
            hc_idx = [i for i in hc_idx if i not in un_idx]
            hc_labels = np_utils.to_categorical(hc_labels, n_classes)
            DH = np.take(DU[0], hc_idx, axis=0), hc_labels

        DU = np.delete(DU[0], un_idx, axis=0), np.delete(DU[1], un_idx, axis=0)

        if i % args.fine_tunning_interval == 0:
            dtrain_x = np.concatenate((DL[0], DH[0]))
            dtrain_y = np.concatenate((DL[1], DH[1]))
            model.fit(dtrain_x, dtrain_y, validation_data=(x_test, y_test), batch_size=args.batch_size,
                      shuffle=False, epochs=args.epochs, verbose=args.verbose, callbacks=[earlystop])
            args.delta -= (args.threshold_decay * args.fine_tunning_interval)

        _, acc = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=args.verbose)

        print(
            'Iteration: %d; High Confidence Samples: %d; Uncertain Samples: %d; Delta: %.5f; Labeled Dataset Size: %d; Accuracy: %.2f'
            % (i, len(hc_idx), len(un_idx), args.delta, len(DL[0]), acc))


if __name__ == '__main__':
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
    parser.add_argument('-epochs', default=5, type=int, help="Number of epoch to train. default: 5")
    parser.add_argument('-batch_size', default=32, type=int, help="Number of samples per gradient update. default: 32")
    parser.add_argument('-chkt_filename', default="ResNet18v2-CIFAR-10_init_ceal.hdf5",
                        help="Model Checkpoint filename to save")
    parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, help="Fine-tuning interval. default: 1")
    parser.add_argument('-T', '--maximum_iterations', default=10, type=int,
                        help="Maximum iteration number. default: 10")
    parser.add_argument('-i', '--initial_annotated_perc', default=0.1, type=float,
                        help="Initial Annotated Samples Percentage. default: 0.1")
    parser.add_argument('-dr', '--threshold_decay', default=0.0033, type=float,
                        help="Threshold decay rate. default: 0.0033")
    parser.add_argument('-delta', default=0.05, type=float,
                        help="High confidence samples selection threshold. default: 0.05")
    parser.add_argument('-K', '--uncertain_samples_size', default=2000, type=int,
                        help="Uncertain samples selection size. default: 2000")
    parser.add_argument('-uc', '--uncertain_criteria', default='ms',
                        help="Uncertain selection Criteria: \'lc\' (Least Confidence), \'ms\' (Margin Sampling), \'en\' (Entropy). default: lc")
    parser.add_argument('-ce', '--cost_effective', default=True,
                        help="whether to use Cost Effective high confidence sample pseudo-labeling. default: True")
    args = parser.parse_args()

    # keras callbacks
    earlystop = EarlyStopping(monitor='val_loss', patience=1)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
    checkpoint = ModelCheckpoint(args.chkt_filename, monitor='val_acc', save_best_only=True)

    run_ceal(args)
