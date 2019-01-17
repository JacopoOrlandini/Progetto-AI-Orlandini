from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import networkx as nx
import time

from numpy.core._multiarray_umath import ndarray
from scipy import sparse



def readCoraGraph(nodesPath="data/cora/cora.content", edgesPath="data/cora/cora.cites"):
    G = nx.Graph()
    orderedNodesList = []
    edges = open(edgesPath, "r")
    nodes = open(nodesPath, "r")
    for n in nodes.readlines():
        nodeID = n.split("\t")[0]
        G.add_node(nodeID)
        orderedNodesList.append(nodeID)
    # print("INITIAL NODES ARE :"+ str(len(G.nodes())))
    for line in edges.readlines():
        fields = line.split("\t")
        fields[1] = fields[1].rstrip('\n')  #delete \n
        G.add_edge(fields[0], fields[1])
    return G, orderedNodesList


def cv_my_shuffle(x, y):
    """
    format new matrix  = X[],Y[], shuffled_index
    :param y: label node, ndarray
    :param x: feature node, numpy.matrix
    :return: shuffle new concatenate matrix XY
    """
    print("\n\nShuffle function :")
    print("\tx =  type{}, shape {}, element type{}".format(type(x), x.shape, type(x[0][0])))
    print("\ty =  type{}, shape {}, type element{}".format(type(y), y.shape, type(y[0][0])))

    concatenation_mat = np.concatenate((x, y), axis=1)
    index = np.arange(len(y)).reshape((-1, 1))
    concatenation_mat = np.append(concatenation_mat, index, axis=1)   # index Y before shuffle
    np.random.shuffle(concatenation_mat)    #row shuffle
    print("\tShape matrix X_Y_I_ =  type{}, shape {}\n\n".format(type(concatenation_mat), concatenation_mat.shape))

    X = concatenation_mat[:, :1433]
    y = concatenation_mat[:, 1433:1440]
    shuffle_index = concatenation_mat[:, -1]
    y_shuffled = np.zeros(y.shape, dtype=np.int32)  # build numpy.ndarray
    for counter, value in enumerate(y):
        y_shuffled[counter] = value
    print("New matrix info :")
    print("\tnew y : {} type = {}".format(y_shuffled.shape, type(y_shuffled)))
    print("\tnew X : {} type = {}".format(x.shape, type(x)))
    print("\tnew I : {}".format(shuffle_index.shape))

    return X, y_shuffled, shuffle_index



def cv_get_slice(y, rate=0.052):
    """
    Implementation of cross validation on kegra model
    :param y: label of graph
    :param rate: label rate of graph
    :return: size fold for cross validation
    """
    print("Cross-validation GET FOLD with rate [{}] on {} nodes".format(rate, len(y)))
    fold_size = int(round(len(y)*rate))
    print("\tFold size (training) = {}\n\n".format(fold_size))
    return fold_size


def cv_get_partition_orig(y, size_fold, index):
    """
    :param y: [ndarray] . y = load_data(...)
    :param size_fold: size of training set
    :param index : index after shuffle , refers to old index (Y0 index)
    :return: same as get_split()
        y_train all zeros.
        for counter in range(cv_fold):
            y_train[shuffled_index] = y[shuffled_index]

    """
    # versione1
    # devo mettere in idx-TRAIN i nuovi valori non semplicemnte i range(140)
    idx_train = range(size_fold)
    idx_val = range(size_fold, int(round(len(y) - size_fold) / 2))
    idx_test = range(int(round(len(y) - size_fold) / 2), len(y))
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)

    # for ii in range(5):
    #     print("shuffle value = {}\t new_i = {}\t old_i = {}".format(y[ii], ii, int(index[ii])))

    for i in idx_train:
        y_train[int(index[i])] = y[int(index[i])]  # look version
        """
        versione 2 faccio entrare la y_ (shuffled_Y) e poi uso idx train per la parte destra mentre index[i] per la sx

        for i in idx_train:
            y_train[int(index[i])] = y_shuffled[i] 
        """
    for j in idx_val:
        y_val[int(index[j])] = y[int(index[j])]
    for k in idx_test:
        y_test[int(index[k])] = y[int(index[k])]
    mask = np.zeros(y.shape[0])
    for l in idx_train:
        mask[int(index[l])] = 1
    print(idx_train)
    print("Y TRAINING : shape = {} ".format(y_train.shape))
    print("Y VALIDATION : shape = {} ".format(y_val.shape))
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, np.array(mask, dtype=np.bool)


def cv_get_partition(y, size_fold, index):
    """
    :param y: [ndarray] . y = load_data(...)
    :param size_fold: size of training set
    :param index : index after shuffle , refers to old index (Y0 index)
    :return: same as get_split()
        y_train all zeros.
        for counter in range(cv_fold):
            y_train[shuffled_index] = y[shuffled_index]

    """
    # versione1
    idx_train = range(size_fold)
    idx_val = range(size_fold, int(round(len(y)-size_fold)/2))
    idx_test = range(int(round(len(y)-size_fold)/2), len(y))
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)

    for i in idx_train:
        y_train[int(index[i])] = y[i]   # look version 2
        """
        versione 2 faccio entrare la y_ (shuffled_Y) e poi uso idx train per la parte destra mentre index[i] per la sx

        for i in idx_train:
            y_train[int(index[i])] = y_shuffled[i] 
        """
    for j in idx_val:
        y_val[int(index[j])] = y[j]
    for k in idx_test:
        y_test[int(index[k])] = y[k]
    mask = np.zeros(y.shape[0])
    for l in idx_train:
        mask[int(index[l])] = 1

    #update indices
    list_y_tr = []
    list_y_v = []
    list_y_te = []
    for i in range(len(idx_train)):
        list_y_tr.append(int(index[i]))
    for i in range(len(idx_val)):
        list_y_v.append(int(index[i]))
    for i in range(len(idx_test)):
        list_y_te.append(int(index[i]))


    print("Y TRAINING : shape = {} ".format(y_train.shape))
    print("Y VALIDATION : shape = {} ".format(y_val.shape))

    return y_train, y_val, y_test, list_y_tr, list_y_v, list_y_te, np.array(mask, dtype=np.bool)



# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'localpool'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience


X, A, y = load_data(dataset=DATASET)

# Cross-validation on single RUN
X_, y_, index_ = cv_my_shuffle(X, y)


cv_size = cv_get_slice(y_)
cake = round(len(y_)/cv_size)    #restituisce 19 fette 19*141 = 2679
print("value size fetta = {}, total fette = {}".format(cv_size, cake))
result = []
for k in range(cake):
    print("\n\nfold {} in {}".format(k, cake))
    y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = cv_get_partition(y_, cv_size, index_)

    X /= X.sum(1).reshape(-1, 1)   #non funziona questo reshape nella loro versione. normalizza e basta.

    if FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        print('Using local pooling filters...')
        A_ = preprocess_adj(A, SYM_NORM)
        support = 1
        graph = [X, A_]
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
        print(G)
    elif FILTER == 'chebyshev':
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A, SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        support = MAX_DEGREE + 1
        graph = [X]+T_k
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

    else:
        raise Exception('Invalid filter type.')

    X_in = Input(shape=(X.shape[1],))

    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

    model = Model(inputs=[X_in] + G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999

    # Fit
    for epoch in range(1, NB_EPOCH+1):
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, sample_weight=train_mask,
                  batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])
        print("\tEpoch: {:04d}".format(epoch),
              "\ttrain_loss= {:.4f}".format(train_val_loss[0]),
              "\ttrain_acc= {:.4f}".format(train_val_acc[0]),
              "\tval_loss= {:.4f}".format(train_val_loss[1]),
              "\tval_acc= {:.4f}".format(train_val_acc[1]),
              "\ttime= {:.4f}".format(time.time() - t))

        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= PATIENCE:
                print('\tEpoch {}: early stopping'.format(epoch))
                break
            wait += 1

    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    print("\nTest set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))

    result.append(test_acc[0])

    # Next slice of cake
    idx_train = range(cv_size)
    idx_val = range(cv_size, int(round(len(y) - cv_size)/2))
    idx_test = range(int(round(len(y)-cv_size)/2), len(y))
    next_k = list(idx_val) + list(idx_test) + list(idx_train)

    # Update y and indices for next iteration with new slice of cake
    y_ = y_[next_k]
    index_ = index_[next_k]

avg = 0
print("Results : \n{}".format(result))
for i in result:
    avg += i
avg = avg/len(result)
print("average = {}".format(avg))


