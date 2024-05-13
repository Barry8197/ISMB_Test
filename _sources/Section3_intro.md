# <font color='darkblue'> Introduction to Graph Neural Networks </font>
Graph Neural Networks (GNN) are a powerful architecture for the learning of graph structure and information in a supervised setting. In this workshop we will implement a Graph Convolutional Network (GCN) model from the [Deep Graph Library](https://www.dgl.ai/) in Python. 

The goal of GNN's is to learn an embedding space for nodes which captures both node feature representation and graph structure. Intuitively, if two nodes are connected and belong to the same class they should be close together in the embedding space. Conversely, if two nodes are connected but do not belong to the same class we want them to be separated in the embedding space. Thus, we cannot rely on graph structure alone and necessitates the requirement to include node feature representation also. The method in which capture this similarity is through the message passing algorithm discussed below. 

<p align ="center"><img src="GNN_Learning.png" alt="drawing" width="600"/></p>

The differentiation between GCN and neural network architectures is their ability to learn from the local neighbourhood as opposed to handcrafted network features. The performance of GCN and other GNN architectures has been demonstrated on a variety of benchmark tasks, hence extending their application to a biomedical setting is an exciting avenue. 

## <font color='darkblue'> Message Passing in GNN's </font>
Message passing is the method of information exchange among nodes. It is performed so that similar nodes are mapped to similar embedding spaces during the learning phase of the GNN. i.e. if A, D and B are cancer patients, we want D and B to exchange this information with A, but we want C to exchange the information that it is not a cancer patient. In doing so, A will see that it is connected to 2 cancer patients and 1 non cancer patient thus, learning that it is more likely also a cancer patient. 

<p align ="center"><img src="GNNMessagePassing.png" alt="drawing" width="1000"/></p>

The message passing algorithm consists of three core steps : Propagation, Aggregation and Update. For a single node "u", the hidden embedding $h(k)_u$ can be formulated as per the general equation, where $N(u)$ is the neighbourhood of u. 

<p align ="center"><img src="MessagePassingalgo.png" alt="drawing" width="1000"/></p>

### <font color='darkblue'> Propagation </font>
The first step of message passing simply involves gathering all embeddings $h(k)_v$ for every node u. During this step it is common to apply an augmentation. In a GCN this augmentation is a Multi Layer Perceptron of arbritrary dimension. In this manner each node receives a single augmented message from its neighbours.

It is important to note that this step makes GNN's invariant to the order of the nodes. i.e. it does not matter if the message ordering to A is D, B, C or B, C, D. As we perform an augmentation on a set of node embeddings, the GNN is unaffected by permutations to the ordering. 

### <font color='darkblue'> Aggregation </font>
The method of node aggregation is a significant differentiator between different GNN architecture. For example, GCN's employ mean pooling whereas, GraphSage employs max-pooling. The aggregation step summarises the information received from the other nodes. 

### <font color='darkblue'> Update </font>
Finally, the node updates it's embedding position based on the new information it has gathered from it's neighbouring nodes. 

## <font color='darkblue'> Graph Convolutional Network (GCN) </font>
In this workshop we will work exclusively with the GCN model shown below. The GCN consists uses Multi Layer Perceptron linear layers for message augmentation, element-wise mean pooling and intermediate ReLU non-linearity. 

<p align ="center"><img src="GCN.png" alt="drawing" width="800"/></p>
<p align ="center"><img src="GCN2.png" alt="drawing" width="800"/></p>

## <font color='darkblue'> Node Classification </font>

## <font color='darkblue'> Deep Graph Library </font>

## <font color='darkblue'> Generation Scotland Dataset </font>