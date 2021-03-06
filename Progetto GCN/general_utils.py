#Author: Gianfranco Lombardo

def minMaxNormalization(X,normalizationAxis=0):
    result = (X - np.min(X, axis=normalizationAxis)) / (np.max(X, axis=normalizationAxis) - np.min(X, axis=normalizationAxis))  

def embedding(G,name,spaceDimension = 32,lenWalk=20,Pparameter=1,Qparameter=0.1,n_walk=20):
    modelName =str(name)+"_"+str(spaceDimension)+"_"+str(lenWalk)+"_"+str(n_walk)+"_P_"+str(Pparameter)+"_Q_"+str(Qparameter)+".model"
    modelFile=Path(modelName)

    if modelFile.is_file():
        model = gensim.models.Word2Vec.load(modelName)
        print("---- Model loaded")
    else:
        print("---- Random walks generation----")
        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(G, dimensions=spaceDimension, walk_length=lenWalk, p=Pparameter, q=Qparameter,
                            num_walks=n_walk, workers=1)
        # Embed
        model = node2vec.fit(window=10, min_count=1,
                             batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        print("Embedding completed !")
        print("---- SAVING MODEL----")
        # Save model for later use
        model.save(modelName)
    return model
    
def readCoraGraph(nodesPath="data/cora/cora.content",edgesPath="data/cora/cora.cites"):
    edges = open(edgesPath, "r")
    G = nx.Graph()
    nodes = open(nodesPath, "r")
    orderedNodesList = []
    for n in nodes.readlines():
        nodeID = n.split("\t")[0]
        G.add_node(nodeID)
        orderedNodesList.append(nodeID)

    # print("INITIAL NODES ARE :"+ str(len(G.nodes())))
    for line in edges.readlines():
        fields = line.split("\t")
        # print("Link between: "+fields[0]+" and "+fields[1].replace("\n",""))
        G.add_edge(fields[0].replace("\t",""),fields[1].replace("\n",""))
    # print("Graph has "+str(len(G.nodes))+ " nodes")
    return G, orderedNodesList


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)  # restituisce un range
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

