from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import networkx as nx
import time

import datetime

def readCoraGraph(dataset_name="cora"):
    nodesPath = "data/"+dataset_name+"/"+dataset_name + ".content"
    edgesPath = "data/"+dataset_name+"/"+dataset_name+".cites"
    G = nx.Graph()
    orderedNodesList = []
    try:
        edges = open(edgesPath, "r")
        nodes = open(nodesPath, "r")
    except IOError:
        print(f"Check dataset name or dataset location in ./data/[{dataset_name}]/[{dataset_name}].*")
    for n in nodes.readlines():
        nodeID = n.split("\t")[0]
        G.add_node(nodeID)
        orderedNodesList.append(nodeID)
    for line in edges.readlines():
        fields = line.split("\t")
        fields[1] = fields[1].rstrip('\n')
        G.add_edge(fields[0], fields[1])
    print(G.number_of_nodes())
    print(G.number_of_edges())
    return G, orderedNodesList


def cv_my_shuffle(features, labels):
    """
    format new matrix  = X[],Y[], shuffled_index

    Descrizione metodo: il metodo prende gli input da load data, concatena (nel seguente ordine)la matrice
    delle features X e la matrice delle Labels Y per poi aggiungere l'ultima colonna che rappresenta l'enumerazione deli nodi.
    La matrice concatenation_mat viene shufflata secondo le righe.
    Successivamente vengono recuperate le nuove matrici X e y e gli indici shufflati dei nodi.
    Per quanto riguarda la X che viene restituita non verrà usata all'interno del train.py in quanto devo mantenere le corrispondenze
    con la matrice di adiacenza A che viene calcolata all'inizio con il metodo load_data.
    :param labels: label node, ndarray
    :param features: feature node, numpy.matrix
    :return: shuffle new concatenate matrix XY
    """
    #TODO tipo dentro i vettori sono tutti diversi attualmente
    index = np.arange(len(labels)).reshape((-1, 1))
    print(f"Descrizione features: X\n\tdimensione: {features.shape}\n\ttipo data: {features.dtype}")
    print(f"Descrizione labels: y\n\tdimensione: {labels.shape}\n\ttipo data: {labels.dtype}")
    print(f"Descrizione indexs: \n\tdimensione: {index.shape}\n\ttipo data: {index.dtype}")
    concatenation_mat = np.concatenate((features, labels), axis=1)
    concatenation_mat = np.append(concatenation_mat, index, axis=1)
    np.random.shuffle(concatenation_mat)
    shuffle_features = concatenation_mat[:, :features.shape[1]]
    y = concatenation_mat[:, features.shape[1]: features.shape[1] + labels.shape[1]]
    shuffle_index = concatenation_mat[:, -1]
    shuffle_labels = np.zeros(y.shape, dtype=np.int32)
    for counter, value in enumerate(y):
        shuffle_labels[counter] = value
    print("\n\nShuffle function... [DONE]")
    return shuffle_features, shuffle_labels, shuffle_index


def cv_get_slice(y, rate=0.052):
    """
    Descrizione metodo: ritorna il numero intero arrotondato della dimensione del set di training del modello.
    :param y: label of graph
    :param rate: label rate of graph
    :return: int , size of y_training
    """
    print("Cross-validation GET K with rate [{}] on {} nodes".format(rate, len(y)))
    fold_size = int(round(len(y)*rate))
    print("Fold size (training) = {}, label_rate = {}... [DONE]".format(fold_size, rate))
    return fold_size


def cv_get_partition(y_shuffle, size_fold, index):
    """
    Descrizione metodo: Metodo per restituire le partizioni di y_train, y_val, y_test.
    La politica adottata è stata:
    1 - la dimensione del train è stata calcolata attraverso il metodo cv_get_slice con la possibilita di usare label_rate
        per fornire la dimensione in percentuale.
    2 - la dimensione di validation e test sono uguali e dividono il dataset meno la parte di training in 2 parti uguali
    3 - Vengono popolate le matrici secondo il vecchio ordinamento di Y0 (ovvero y iniziale della load ) in modo da preservare le relazioni
    con la X e la matrice di Adiacenza A.
    4 - IMPORTANTE - notare che il metodo implementato da kegra "EVALUATE_PREDS" impone che in idx_train, idx_val, idx_test
        escano delle liste (o range) in cui sono effettivamente presenti gli indici che siamo andati a popolare nelle matrici di riferimento y_*.
        Nel momento che non usiamo pià una distribuzione di label come da loro proposto, devo andare ad aggiornare le matrici
        con i nuovi indici associati dopo la shufflata.     new_idx_train , new_idx_val, new_idx_test.
        Sono le liste di dimensione uguale a quante label ho messo nei vari Y_* con gli indici shufflati.
        Anche la mask viene modificata in base agli indici shufflati.

    :param y: [ndarray] . shuffled_Y, perciò y[i] corrisponde al valore onehot_encode dell'elemento nella prima posizione nella matrice shufflata.
    :param size_fold: size of training set
    :param index : index after shuffle. In index[i] quindi trovo l'indice rispetto alla matrice Y0.
    :return: same as get_split with updated index and value.
    """
    idx_train = range(size_fold)
    idx_val = range(size_fold, int(round(len(y)-size_fold)/2))
    idx_test = range(int(round(len(y)-size_fold)/2), len(y))
    y_train = np.zeros(y_shuffle.shape, dtype=np.int32)
    y_val = np.zeros(y_shuffle.shape, dtype=np.int32)
    y_test = np.zeros(y_shuffle.shape, dtype=np.int32)
    # fill y set
    for i in idx_train:
        '''
        # Cosa faccio? Y0 = [1:a, 2:b, 3:c, 4:d] --> shuffle --> Y1 = [3:c, 1:a, 4:d, 2:b]
        
        Mettiamo il caso iniziale : 
            - y_train viene inizializzata a zero
            - i = 0
            - index si riferisce agli indici shufflati
        Quindi: 
            1)index[0] = 3            \
                                      | ---> da (1) e (2)    --->    y_train[3] = c 
            2)Y1[0] = c               /      
        
        '''
        y_train[int(index[i])] = y_shuffle[i]
    for j in idx_val:
        y_val[int(index[j])] = y_shuffle[j]
    for k in idx_test:
        y_test[int(index[k])] = y_shuffle[k]
    mask = np.zeros(y_shuffle.shape[0])
    for l in idx_train:
        mask[int(index[l])] = 1
    # update indices
    new_idx_train = []
    new_idx_val = []
    new_idx_test = []
    for i in idx_train:
        new_idx_train.append(int(index[i]))     # metto gli indici associati a y_train per valutarli successivamente
    for i in idx_val:
        new_idx_val.append(int(index[i]))
    for i in idx_test:
        new_idx_test.append(int(index[i]))
    print("\tGet_partition... [DONE]")
    return y_train, y_val, y_test, new_idx_train, new_idx_val, new_idx_test, np.array(mask, dtype=np.bool)

# Define parameters


DATASET = 'cora'
FILTER = 'localpool'  # 'localpool'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
PATIENCE = 10  # early stopping patience

# MY MACRO
PATH = "data/"+DATASET+'/'
RATE = 0.052
NB_EPOCH = 200
RUN_TOT = 100

# Opening file
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
file = open("cora_"+st, "w")

end_average = 0 # average on 100 run
list_run_result = []    # list of single run average

X, A, y = load_data(PATH, DATASET)
for run_id in range(RUN_TOT):
    # Cross-validation on single RUN
    X_, y_, index_ = cv_my_shuffle(X, y)    # every run has 1 shuffle on dataset
    cv_size = cv_get_slice(y_, rate=RATE)
    K_TOT = round(len(y_)/cv_size)    # number of slice in cake(dataset) of cross validation
    if run_id == 0:
        file.write("Total Run: " + str(RUN_TOT) + "\n")
        file.write("PATIENCE: " + str(PATIENCE) + "\n")
        file.write("FILTER: " + FILTER + "\n")
        file.write("EPOCH total: "+ str(NB_EPOCH) + "\n")
        file.write("size Label Y: "+ str(len(y_)) + "\n")
        file.write("label rate: "+ str(RATE) + "\n")
        file.write("size training set: "+ str(cv_size) + "\n")
        file.write("size K_TOT: "+ str(K_TOT) + "\n")
        print("Search legend for results (ctrl+f + insert_code)", file=file)
        print("\t - to find specific run [code]: R{run_id} (example R1)", file=file)
        print("\t - to find specific run Epoch list results [code]: R{run_id}E (example R1E)", file=file)
        print("\t - to find specific run Epoch average result [code]: R{run_id}AE (example R4AE)", file=file)
        print("\t - to show all run average epochs results: A_all", file=file)
        print("\t - to show average on all run: A_end", file=file)
    file.write("\nCurrent Run: " + str(run_id) + " of " + str(RUN_TOT) + "\n")
    result = []
    for k in range(K_TOT):
        print("\n\n###  CROSS VALIDATION    k = {} in {} ###".format(k, K_TOT))
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = cv_get_partition(y_, cv_size, index_)
        X /= X.sum(1).reshape(-1, 1)

        if FILTER == 'localpool':
            """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
            print('\tUsing local pooling filters...')
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

        # Next slice of K_TOT
        idx_train = range(cv_size)
        idx_val = range(cv_size, int(round(len(y) - cv_size) / 2))
        idx_test = range(int(round(len(y) - cv_size) / 2), len(y))
        next_k = list(idx_val) + list(idx_test) + list(idx_train)

        # Update y and indices for next iteration with new slice of K_TOT
        y_ = y_[next_k]
        index_ = index_[next_k]
    avg = 0
    print(f"\tR{run_id}E: \n\t{result}")  #result = all 200 (epochs) test.acc on single run
    print(f"\tR{run_id}E: \n\t{result}", file=file)  # result = all 200 (epochs) test.acc on single run
    for i in result:
        avg += i
    avg = avg/len(result)
    list_run_result.append(avg) # list run = all average_result of run
    print(f"\tR{run_id}AE: \n\t{avg}")
    print(f"\tR{run_id}AE: \n\t{avg}", file=file)

for i in list_run_result:
    end_average += i
end_average = end_average/len(list_run_result)
print(f"\n\nA_all: \n{list_run_result}")
print(f"Avarage result on {RUN_TOT} iteration is : {end_average}")
print(f"\n\nA_all: \n{list_run_result}", file=file)
print(f"\n\nA_end (average on {RUN_TOT} run) is: {end_average}", file=file)