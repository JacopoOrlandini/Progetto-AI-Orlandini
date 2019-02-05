import numpy as np
def data_shuffle(features, labels):
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
    index = np.arange(len(labels)).reshape((-1, 1))
    # print(f"Descrizione features: X\n\tdimensione: {features.shape}\n\ttipo data: {features.dtype}")
    # print(f"Descrizione labels: y\n\tdimensione: {labels.shape}\n\ttipo data: {labels.dtype}")
    # print(f"Descrizione indexs: \n\tdimensione: {index.shape}\n\ttipo data: {index.dtype}")
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
    
def get_fold_size(y, rate):
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

def get_cross_validation_folds(y,y_shuffle, size_fold, index):
    """
    Descrizione metodo: Metodo per restituire le partizioni di y_train, y_val, y_test.
    La politica adottata è stata:
    1 - la dimensione del train è stata calcolata attraverso il metodo get_fold_size con la possibilita di usare label_rate
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
        '''
        Se training set è composto da 2 elementi, val da 1 e test da 1 allora:
        new_idx_train = [3, 1]
        new_idx_val = [4]
        new_idx_test = [2]
        '''
        new_idx_train.append(int(index[i]))     # metto gli indici associati a y_train per valutarli successivamente
    for i in idx_val:
        new_idx_val.append(int(index[i]))
    for i in idx_test:
        new_idx_test.append(int(index[i]))
    print("\tGet_partition... [DONE]")
    return y_train, y_val, y_test, new_idx_train, new_idx_val, new_idx_test, np.array(mask, dtype=np.bool)