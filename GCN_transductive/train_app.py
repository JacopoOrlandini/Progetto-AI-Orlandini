"""

__version__ = 1.3
__data__ = 09/02/2019
__author__ = Jacopo Orlandini


'''    Artificial Intelligence     '''

"""
import argparse
import datetime
import os
import time
from os import listdir
from os.path import isfile, join

from kegra.layers.graph import GraphConvolution
from kegra.utils import *
from keras.backend import clear_session
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import general_utils as gu
import trasductive_cross_validation as tp_cv

'''SET UP PARAM PROJECT '''
def arg_parsing():
    # flag set in args.log
    global args
    parser = argparse.ArgumentParser(description="Progetto GCN cross validation")
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()
    return args
def set_project_parameters():
    global FILTER, MAX_DEGREE, SYM_NORM, PATIENCE, PATH, RATE, NB_EPOCH, RUN_TOT, Cycle_inner_Epoch, DATASET, FILENAME
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H_%M_%S')
    # Define parameters
    DATASET = 'cora'
    FILTER = 'localpool'  # 'localpool'
    MAX_DEGREE = 2  # maximum polynomial degree
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    PATIENCE = 10  # early stopping patience
    PATH = "data/"+DATASET+'/'
    RATE = 0.052    #percentuale per training set
    NB_EPOCH = 200
    RUN_TOT = 100
    Cycle_inner_Epoch = 5
    FILENAME = DATASET + "_" + st
    if os.path.isdir(PATH):
        print("Found folder: " + PATH)
        print("Found file: " + str([f for f in os.listdir(PATH) if f.find(DATASET) >= 0]))
        if os.access(PATH, os.W_OK):
            print("Writable file: GRANTED")
        else:
            print("Writable: DENIED")
            exit(-1)

    else:
        print("Loading files error. Copy dataset !!!")
        exit(-1)
def set_project_parameters_emb():
    global FILTER, MAX_DEGREE, SYM_NORM, PATIENCE, PATH, RATE, NB_EPOCH, RUN_TOT, Cycle_inner_Epoch, DATASET, FILENAME
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H_%M_%S')
    # Define parameters
    DATASET = 'cora'
    FILTER = 'localpool'  # 'localpool'
    MAX_DEGREE = 2  # maximum polynomial degree
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    PATIENCE = 10  # early stopping patience
    PATH = "data/"+DATASET+'/'
    RATE = 0.052    #percentuale per training set
    NB_EPOCH = 200
    RUN_TOT = 1
    Cycle_inner_Epoch = 1
    FILENAME = DATASET + "_" + st
    if os.path.isdir(PATH):
        print("Found folder: " + PATH)
        print("Found file: " + str([f for f in os.listdir(PATH) if f.find(DATASET) >= 0]))
        if os.access(PATH, os.W_OK):
            print("Writable file: GRANTED")
        else:
            print("Writable: DENIED")
            exit(-1)

    else:
        print("Loading files error. Copy dataset !!!")
        exit(-1)
def custom_load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels, idx_map

'''TRAINING AND VALIDATION'''
def pre_process_data(X, A):
    '''variable after processing will get postfix '_'
        E.g X -> X_
        Use for tkipf version
    '''
    A_ = preprocess_adj(A, SYM_NORM)
    # Normalize X
    X_ = X/X.sum(1).reshape(-1, 1)
    return A_, X_
def cross_validation(X_, y, A_,run_id, file, list_run_result):
    # Cross-validation of a single RUN
    index_ = tp_cv.data_shuffle_fast(y)    # every run has 1 shuffle on data set
    fold_size = tp_cv.get_fold_size(y, rate=RATE)   #length of training set
    K_TOT = round(len(y)/fold_size)    # number of slice in data set due to the cross validation
    file = open(FILENAME, "a")
    result = []     #sono i risultati delle media su 10 iterazioni per fetta k su dati
    #Cycle on k-fold in 0,..,KTOT
    for k in range(K_TOT):
        if args.log :
            print("\n\n###  CROSS VALIDATION    k = {} in {} ###".format(k, K_TOT))
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = tp_cv.get_cross_validation_folds_fast(y, fold_size, index_)
        # Fit
        inner_loop_epoch = []
        inner_loop_epoch_avg = 0
        # inner loop on slice
        for _ in range(Cycle_inner_Epoch):     # ciclo Cycle_inner_epoch volte per ottenere la media sulla fetta
            t_loss, t_acc = tkipf_training(X_, A_, y, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask)
            inner_loop_epoch.append(t_acc)
        for i in inner_loop_epoch:
            inner_loop_epoch_avg += i
        inner_loop_epoch_avg = inner_loop_epoch_avg / len(inner_loop_epoch)
        result.append(inner_loop_epoch_avg)

        # Next slice of K_TOT
        idx_train = range(fold_size)
        idx_val = range(fold_size, int(round(len(y) - fold_size) / 2))
        idx_test = range(int(round(len(y) - fold_size) / 2), len(y))
        next_k = list(idx_val) + list(idx_test) + list(idx_train)

        # Update indices for next iteration with new slice of K_TOT
        index_ = index_[next_k]

    avg = 0
    file.write("R{}K:{}".format(run_id, result)+"\n")
    for i in result:
        avg += i
    avg = avg/len(result)
    list_run_result.append(avg) # list run = all average_result of run
    file.write("R{}AK: {}".format(run_id, avg)+"\n\n")
    file.close()
    return list_run_result
def tkipf_training(X_, A_, y, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask):
    # Define model architecture
    # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
    # This is somewhat hacky, more elegant options would require rewriting the Layer base class.

    # Choosing filter
    if FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        if args.log:
            print('\tUsing local pooling filters...')
        support = 1
        graph = [X_, A_]
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    elif FILTER == 'chebyshev':
        """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
        #ATTENZIONE A globale e non preprocessata
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A, SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        support = MAX_DEGREE + 1
        graph = [X_] + T_k
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
    else:
        raise Exception('Invalid filter type.')

    # Model architecure
    X_in = Input(shape=(X_.shape[1],))
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
    model = Model(inputs=[X_in] + G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

    # Helper variables for k_fold loop
    wait = 0
    preds = None
    best_val_loss = 99999
    t_fit = time.time()
    for epoch in range(1, NB_EPOCH + 1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, sample_weight=train_mask,
                  batch_size=A_.shape[0], epochs=1, shuffle=False, verbose=0)
        if epoch == 1 and args.log:
            print("tempo per fit [1° epoch] = "+ str(time.time()-t))
            print("Pesi iniziali [1° epoch]")
            print(model.get_weights()[0][0])

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A_.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])
        if args.log:
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
            if wait >= PATIENCE :
                if args.log:
                    print('\tEpoch {}: early stopping'.format(epoch))
                break
            wait += 1

    # Testing
    if args.log:
        print("Pesi finali [1° epoch]")
        print(model.get_weights()[0][0])
        print("time to complete fit : " + str(time.time() - t_fit))

    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
    if args.log:
        print("\nTest set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))
    clear_session() #Pulisco la sessione tensorflow --> fix per il tempo incrementale sul fit.
    return test_loss[0], test_acc[0]

'''WRITE FUNCTIONS'''
def write_info(file):
    file.write("Total Run: " + str(RUN_TOT) + "\n")
    file.write("PATIENCE: " + str(PATIENCE) + "\n")
    file.write("FILTER: " + FILTER + "\n")
    file.write("EPOCH total: " + str(NB_EPOCH) + "\n")
    file.write("label rate: " + str(RATE) + "\n")
    file.write("inner loop on fetta: " + str(Cycle_inner_Epoch)+"\n")
    print("Search legend for results (ctrl+f + insert_code)", file=file)
    print("\t - to find specific run [code]: R{run_id} (example R1)", file=file)
    print("\t - to find specific run on fette list results [code]: R{run_id}K (example R1K)", file=file)
    print("\t - to find specific run Epoch average result [code]: R{run_id}AK (example R4AK)", file=file)
    print("\t - to show all run average epochs results: A_all", file=file)
    print("\t - to show average on all run: A_end", file=file)
def write_end_results(list_run_result):
    end_average = 0 # average on 100 run
    for i in list_run_result:
        end_average += i
    end_average = end_average/len(list_run_result)

    #Write results on file
    file = open(FILENAME, "a")
    file.write("\n\nA_all: {}".format(list_run_result))
    file.write("\n\nA_end (average on {} run) is: {}".format(RUN_TOT, end_average))
    file.close()


'''FEATURE SCALING'''
def rescaling(matrix):
    #Feature scaling min-max
    for ii in range(matrix.shape[1]):
        min_max_scaler = preprocessing.MinMaxScaler()
        v = matrix[:, ii]
        v_scaled = min_max_scaler.fit_transform(v)
        matrix[:, ii] = v_scaled
    return matrix
def standardization(matrix):
    # Create the Scaler object
    scaler = StandardScaler()
    # Fit your data on the scaler object
    matrix = scaler.fit_transform(matrix)
    return matrix


'''TASKS'''
def main_CV(args):
    set_project_parameters()
    global X, A     #shared among functions

    X, A, y = load_data(PATH, DATASET)
    A_, X_ = pre_process_data(X, A)  # same A and X for all

    # Opening file
    file = open(FILENAME, "w+")
    if args.log:
        write_info(file)
    file.close()

    accuracy_all_run = []    # list of average accuracy on runs
    for run in tqdm(range(RUN_TOT), desc="cross-validation", total=100):
        accuracy_all_run = cross_validation(X_, y, A_, run, file, accuracy_all_run)
        print("RUN {} DONE".format(run))
    
    #Write final average 
    write_end_results(accuracy_all_run)
def main_EMB(args):
    set_project_parameters_emb()
    global X, A  # shared among functions

    # List all files (find just embedding)
    embedding_files = [f for f in listdir("./embeddings/") if isfile(join("embeddings/", f)) and f.find(DATASET) >= 0]

    X, A, y, mapping = custom_load_data(PATH, DATASET)
    A_ = preprocess_adj(A, SYM_NORM)
    for jj, value in tqdm(enumerate(embedding_files), desc="Embedding"):
        # Create embedding matrix
        wv = gu.load_model(join("embeddings/", embedding_files[jj]))    #load model with word2vec
        
        embedding = []

        # Populate numpy matrix
        for i, nodeid in enumerate(mapping):
            result = gu.get_nodeID_emb(nodeid, wv)
            embedding.append(result)
        matEmbedding = np.matrix(embedding)

        # Normalize embedding matrix and create matEmbedding_
        matEmbedding_ = rescaling(matEmbedding)
        # matEmbedding_ = standardization(matEmbedding)

        X_emb = np.concatenate((X, matEmbedding_), axis=1)

        # Training section
        file = open(FILENAME, "a")
        file.write("{}/{} embedding: {}\n".format(jj, len(embedding_files), embedding_files[jj]))
        file.close()
        accuracy_all_run = []
        for run in tqdm(range(RUN_TOT), desc="cross-validation"):
            accuracy_all_run = cross_validation(X_emb, y, A_, run, file, accuracy_all_run)



if __name__ == "__main__":
    '''MAIN SECTION
        To run with log use python train.py -log
        To run with no log use python train.py

        -   CV  : for cross-validation      {RUN_TOT=100, EPOCH=200, CYCLE=5},  set_project_parameters ,    
        -   EMB : for embedding accuracy    {RUN_TOT=1, EPOCH=200, CYCLE=1},    set_project_parameters_emb, 
        Change in __main__ to change performing task 
    '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # tensorflow verbosity
    args = arg_parsing()
    main_EMB(args)
