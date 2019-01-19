from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from kegra.layers.graph import GraphConvolution
from kegra.utils import *
import networkx as nx
from scipy import sparse


import time


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


def cv_my_shuffle(y, x):
    """
    format new matrix  = y[],X[],int_index
    :param y: label node, ndarray (need to cast to np.matrix for concatenation)
    :param x: feature node, numpy.matrix
    :return: shuffle new concatenate matrix XY
    """
    print("\n\nShuffle function :")
    print("\tx =  type{}, shape {}".format(type(x), x.shape))
    print("\ty =  type{}, shape {}".format(type(y), y.shape))
    #y = np.matrix(y)
    print("\tcasting y =  type{}, shape {}".format(type(y), y.shape))
    new_xy = np.concatenate((y, x), axis=1)
    #concatenation_mat = np.concatenate((concatenation_mat, A.todense()), axis=1)
    print(new_xy.shape)
    print(type(y))
    #new_xy = np.concatenate((new_xy, A.todense()), axis=1)
    #index = np.arange(len(y)).reshape((-1, 1))
    #new_xy = np.append(new_xy, index, axis=1)   # ultima colonna sono i corrispondenti al node X

    np.random.shuffle(new_xy)
    print("\tShape matrix_YX =  type{}, shape {}\n\n".format(type(new_xy), new_xy.shape))
    X = new_xy[:, 7:1440]

    return X, y


def cv_get_fold(y, rate=0.052):
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

def cv_get_partition(y, size_fold):
    idx_train = range(size_fold)
    idx_val = range(size_fold, int(round(len(y)-size_fold)/2))
    idx_test = range(int(round(len(y)-size_fold)/2), len(y))
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask



# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'localpool'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience

# X = csr_matrix.
# A = symmetric adjacency matrix, coo_matrix = A sparse matrix in COOrdinate format.
# Y = np.array di one_hot encoding

X, A, y = load_data(dataset=DATASET)


# Cross-validation on single RUN
X, y = cv_my_shuffle(y, X)
#A = conc_mat[:, 1440:]
#A = sparse.csr_matrix(A)


print("New matrix info :")
print("\ty : {}".format(y.shape))
print("\tX : {}".format(X.shape))
cv_size = cv_get_fold(y)    # ritorna dimensione del fold in base al rate inserito (default = 0.052)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = cv_get_partition(y, cv_size)

#y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)


# Normalize X
X /= X.sum(1).reshape(-1, 1)     # provide 1 column but unkown row. vertical vector, normalitation on row sum.

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

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

# Compile model
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)    # y_train.shape = (2708, 7)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
