from __future__ import print_function


from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import networkx as nx
import time
import trasductive_cross_validation as tp_cv
import datetime

OUTPUT_ENABLED = True

"""

This version is suitable for python 3.5 and next versions

"""


# def readCoraGraph(dataset_name="cora"):
#     nodesPath = "data/"+dataset_name+"/"+dataset_name + ".content"
#     edgesPath = "data/"+dataset_name+"/"+dataset_name+".cites"
#     G = nx.Graph()
#     orderedNodesList = []
#     try:
#         edges = open(edgesPath, "r")
#         nodes = open(nodesPath, "r")
#     except IOError:
#         # print(f"Check dataset name or dataset location in ./data/[{dataset_name}]/[{dataset_name}].*")
#     for n in nodes.readlines():
#         nodeID = n.split("\t")[0]
#         G.add_node(nodeID)
#         orderedNodesList.append(nodeID)
#     for line in edges.readlines():
#         fields = line.split("\t")
#         fields[1] = fields[1].rstrip('\n')
#         G.add_edge(fields[0], fields[1])
#     print(G.number_of_nodes())
#     print(G.number_of_edges())
#     return G, orderedNodesList











def gcn_training(X, G, graph):

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
    for epoch in range(1, NB_EPOCH + 1):
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, sample_weight=train_mask,
                  batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])
        '''print("\tEpoch: {:04d}".format(epoch),
               "\ttrain_loss= {:.4f}".format(train_val_loss[0]),
               "\ttrain_acc= {:.4f}".format(train_val_acc[0]),
               "\tval_loss= {:.4f}".format(train_val_loss[1]),
               "\tval_acc= {:.4f}".format(train_val_acc[1]),
               "\ttime= {:.4f}".format(time.time() - t))
        '''
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
    return test_loss[0], test_acc[0]


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


Cycle_inner_Epoch = 5


# Opening file
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H_%M_%S')
file = open(DATASET+"_" + str(st), "w+")

end_average = 0 # average on 100 run
list_run_result = []    # list of single run average

X, A, y = load_data(PATH, DATASET)
for run_id in range(RUN_TOT):
    # print(f"Current run: {run_id}")
    print("Current Run : {}".format(run_id))
    # Cross-validation on single RUN
    X_, y_, index_ = tp_cv.data_shuffle(X, y)    # every run has 1 shuffle on data set
    fold_size = tp_cv.get_fold_size(y_, rate=RATE)
    K_TOT = round(len(y_)/fold_size)    # number of slice in data set due to the cross validation
    # Legend and Info about model and parameters
    if run_id == 0:
        file.write("Total Run: " + str(RUN_TOT) + "\n")
        file.write("PATIENCE: " + str(PATIENCE) + "\n")
        file.write("FILTER: " + FILTER + "\n")
        file.write("EPOCH total: " + str(NB_EPOCH) + "\n")
        file.write("size Label Y: " + str(len(y_)) + "\n")
        file.write("label rate: " + str(RATE) + "\n")
        file.write("size training set: " + str(fold_size) + "\n")
        file.write("size K_TOT: " + str(K_TOT) + "\n")
        file.write("inner loop on fetta: " + str(Cycle_inner_Epoch)+"\n")
        print("Search legend for results (ctrl+f + insert_code)", file=file)
        print("\t - to find specific run [code]: R{run_id} (example R1)", file=file)
        print("\t - to find specific run on fette list results [code]: R{run_id}K (example R1K)", file=file)
        print("\t - to find specific run Epoch average result [code]: R{run_id}AK (example R4AK)", file=file)
        print("\t - to show all run average epochs results: A_all", file=file)
        print("\t - to show average on all run: A_end", file=file)
        file.close()
    file = open(DATASET+"_" + str(st), "a")    
    file.write("\nCurrent Run: " + str(run_id) + " of " + str(RUN_TOT) + "\n")
    result = []     #sono i risultati delle media su 10 iterazioni per fetta k su dati

    for k in range(K_TOT):
        print("\n\n###  CROSS VALIDATION    k = {} in {} ###".format(k, K_TOT))
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = tp_cv.get_cross_validation_folds(y,y_, fold_size, index_)
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

        # Fit
        inner_loop_epoch = []
        inner_loop_epoch_avg = 0
        # inner loop on slice
        for _ in range(Cycle_inner_Epoch):     # ciclo Cycle_inner_epoch volte per ottenere la media sulla fetta
            #file.write(str(_))
            t_loss, t_acc = gcn_training(X, G, graph)
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

        # Update y and indices for next iteration with new slice of K_TOT
        y_ = y_[next_k]
        index_ = index_[next_k]

    avg = 0
    # print(f"\tR{run_id}E: \n\t{result}")  #result = all 200 (epochs) test.acc on single run
    # print(f"\tR{run_id}E: \n\t{result}", file=file)  # result = all 200 (epochs) test.acc on single run
    file.write("\nR{}K:{}".format(run_id, result))
    for i in result:
        avg += i
    avg = avg/len(result)
    list_run_result.append(avg) # list run = all average_result of run
    # print(f"\tR{run_id}AE: \n\t{avg}")
    # print(f"\tR{run_id}AE: \n\t{avg}", file=file)
    file.write("\nR{}AK: {}".format(run_id, avg))
    file.close()

for i in list_run_result:
    end_average += i
end_average = end_average/len(list_run_result)
# print(f"\n\nA_all: \n{list_run_result}")
# print(f"Avarage result on {RUN_TOT} iteration is : {end_average}")
# print(f"\n\nA_all: \n{list_run_result}", file=file)
file = open(DATASET+"_" + str(st), "a")  
file.write("\n\nA_all: {}".format(list_run_result))
# print(f"\n\nA_end (average on {RUN_TOT} run) is: {end_average}", file=file)
file.write("\n\nA_end (average on {} run) is: {}".format(RUN_TOT, end_average))
file.close()