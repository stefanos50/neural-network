from random import randrange,uniform
from matplotlib import interactive
from pip._vendor.distlib.compat import raw_input
interactive(True)
import datetime
import sqlite3
from sklearn import preprocessing
import numpy as np
import math
from numpy import dot
from sklearn.model_selection import KFold, train_test_split

#Connection with the database
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn
#Run SQL Queries
def Make_Query(conn,query,par1,par2):
    cur = conn.cursor()
    if par1 == "" and par2 == "":
        cur.execute(query)
    else:
        cur.execute ("""
    SELECT team_api_id, date, buildUpPlaySpeed, buildUpPlayPassing, chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting, defencePressure, defenceAggression, defenceTeamWidth
    FROM Team_Attributes
    WHERE team_api_id = ?
    AND date = ?""", (int(par1), datetime.datetime.strptime(str(par2), '%Y-%m-%d %H:%M:%S')))
    rows = cur.fetchall()
    results = []
    for row in rows:
        results.append(row)
    if len(results)==1:
        return results[0]
    return results

#Create the vector φ(m)
def neural_vectors(data,labels):
    vector_f = []
    output = []
    for i in range(len(data)):
        team_attributes_H = Make_Query(create_connection("database.sqlite"),"",data[i][0],data[i][16])
        team_attributes_A = Make_Query(create_connection("database.sqlite"),"",data[i][1],data[i][16])
        if len(team_attributes_H)>0 and len(team_attributes_A)>0:
            vector_f.append([team_attributes_H[2],team_attributes_H[3],team_attributes_H[4],team_attributes_H[5],team_attributes_H[6],team_attributes_H[7],team_attributes_H[8],team_attributes_H[9],
                            team_attributes_A[2],team_attributes_A[3],team_attributes_A[4],team_attributes_A[5],team_attributes_A[6],team_attributes_A[7],team_attributes_A[8],team_attributes_A[9],
                            data[i][4], data[i][5], data[i][6],data[i][7], data[i][8], data[i][9],data[i][10], data[i][11], data[i][12],data[i][13], data[i][14], data[i][15]])
            output.append(labels[i])
    return vector_f,output

#If the match result is H then the label is [1,0,0]
#If A then [0,0,1]
#If D then [0,1,0]
def Data_Labels(data):
    match_labels = []
    for i in range(len(data)):
        if data[i][2] > data[i][3]:
            match_labels.append([1, 0, 0])
        elif data[i][2] < data[i][3]:
            match_labels.append([0, 0, 1])
        else:
            match_labels.append([0, 1, 0])
    return match_labels

#Returns the number of the hidden layers that the
#NN needs given the number of inputs,outputs,testing vectors
#and a random number 'a' between 2-10 range
def Num_of_hidden_layers(ns,ni,no,a):
    return math.ceil((ns/(a*(ni+no))))

#ReLu activation function
#Given an array for each element
#if it has a lower value than 0 then set it to 0
#if it has a higher value than 0 then keep it as it is
def relu_activation(data_array):
    return np.maximum(data_array, 0)

#Softmax activation function keeps the output of the NN to be in range 0-1
#and sum to 1
def softmax(z):
    z = z.tolist()
    result = []
    for i in range(len(z)):
        shiftz = z[i] - np.max(z[i])
        exps = np.exp(shiftz)
        result.append(exps / np.sum(exps))
    return np.array(result)

#Calculate the loss of the NN based on the real labels
def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    loss = []
    for i in range(len(softmax_probs_array)):
        loss.append(-np.sum(y_onehot[i] * np.log(softmax_probs_array[i])))
    sum = 0
    for i in range(len(loss)):
        sum += loss[i]
    return sum / len(loss)

#Calculates the accuracy of the NN after the training
#(number of correct NN guesses / total number vectors tested) * 100%
def accuracy(test_data,layer1_weights_array,layer1_biases_array,layer2_weights_array,layer2_biases_array,labels):
    predictions = Neural_Network_Think(test_data,layer1_weights_array,layer1_biases_array,layer2_weights_array,layer2_biases_array)
    preds_correct_boolean =  np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    print("-----------------------------------------------------")
    print("The accuracy of the test is: ",accuracy,"%")
    print("-----------------------------------------------------")
    return accuracy

def regularization_L2_softmax_loss(reg_lambda, weight1, weight2):
    weight1_loss = 0.5 * reg_lambda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * reg_lambda * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss

#Given an input calculate the output of the NN
def Neural_Network_Think(data,layer1_weights_array,layer1_biases_array,layer2_weights_array,layer2_biases_array):
    input_layer = np.dot(data, layer1_weights_array)
    hidden_layer = relu_activation(input_layer + layer1_biases_array)
    scores = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
    probs = softmax(scores)
    return probs

#Given an output and its real pair label
#print if the NN guessed correct or false
def Print_Result(prob,test_data_y):
    prob = prob.tolist()
    test_data_y = test_data_y.tolist()
    print("The NN guessed: ", [float(prob[0]), float(prob[1]), float(prob[2])])
    print("Probability of being H:",float(prob[0]),"%")
    print("Probability of being D:",float(prob[1]),"%")
    print("Probability of being A:",float(prob[2]),"%")
    print("The real result is: ", test_data_y)
    if test_data_y[0] == 1:
        print("It belongs to class H")
    elif test_data_y[1] == 1:
        print("It belongs to class D")
    else:
        print("It belongs to class A")
    if prob.index(max(prob)) == test_data_y.index(max(test_data_y)):
        print("The NN guessed correct.")
    else:
        print("The NN guessed wrong.")

def MNN(vectors,labels):
    f = np.array(vectors)
    lb = np.array(labels)
    kf = KFold(n_splits=10)
    folds_accuracy = []
    folds_l1_w = []
    folds_l1_bias = []
    folds_l2_w = []
    folds_l2_bias = []
    test_sets = []
    for train_index, test_index in kf.split(vectors):
        train_data_x, test_data_x = np.array(f[train_index]),np.array(f[test_index])
        train_data_y, test_data_y = np.array(lb[train_index]),np.array(lb[test_index])
        print("The NN needs ",Num_of_hidden_layers(len(train_data_x),3,28,2)," hidden layer")

        hidden_nodes = 30
        num_labels = train_data_y.shape[1]
        num_features = train_data_x.shape[1]
        learning_rate = .01
        reg_lambda = .01
        # Weights and Bias Arrays
        layer1_weights_array = np.random.normal(0, 1, [num_features, hidden_nodes])
        layer2_weights_array = np.random.normal(0, 1, [hidden_nodes, num_labels])
        layer1_biases_array = np.zeros((1, hidden_nodes))
        layer2_biases_array = np.zeros((1, num_labels))

        #train 50000 times
        for step in range(50000):
            input_layer = np.dot(train_data_x, layer1_weights_array)
            hidden_layer = relu_activation(input_layer + layer1_biases_array)
            scores = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
            output_probs = softmax(scores)

            loss = cross_entropy_softmax_loss_array(output_probs, train_data_y)
            loss += regularization_L2_softmax_loss(reg_lambda, layer1_weights_array, layer2_weights_array)

            output_error_signal = (output_probs - train_data_y) / output_probs.shape[0]

            error_signal_hidden = np.dot(output_error_signal, layer2_weights_array.T)
            error_signal_hidden[hidden_layer <= 0] = 0

            gradient_layer2_weights = hidden_layer.T.dot(output_error_signal)
            gradient_layer2_bias = np.sum(output_error_signal, axis=0, keepdims=True)

            gradient_layer1_weights = train_data_x.T.dot(error_signal_hidden)
            gradient_layer1_bias = np.sum(error_signal_hidden, axis=0, keepdims=True)

            gradient_layer2_weights += reg_lambda * layer2_weights_array
            gradient_layer1_weights += reg_lambda * layer1_weights_array

            layer1_weights_array -= learning_rate * gradient_layer1_weights
            layer1_biases_array -= learning_rate * gradient_layer1_bias
            layer2_weights_array -= learning_rate * gradient_layer2_weights
            layer2_biases_array -= learning_rate * gradient_layer2_bias
            if step % 5000 == 0:
                print
                print('Loss at step {0}: {1}'.format(step, loss))
        prob = Neural_Network_Think(test_data_x, layer1_weights_array,layer1_biases_array, layer2_weights_array, layer2_biases_array)
        for j in range(len(prob)):
            Print_Result(prob[j],test_data_y[j])
        folds_accuracy.append(accuracy(test_data_x,layer1_weights_array,layer1_biases_array,layer2_weights_array,layer2_biases_array,test_data_y))
        folds_l1_w.append(layer1_weights_array)
        folds_l1_bias.append(layer1_biases_array)
        folds_l2_w.append(layer2_weights_array)
        folds_l2_bias.append(layer2_biases_array)
        test_sets.append(test_data_y)
    max_accuracy_index = folds_accuracy.index(max(folds_accuracy))
    return folds_l1_w[max_accuracy_index],folds_l1_bias[max_accuracy_index],folds_l2_w[max_accuracy_index],folds_l2_bias[max_accuracy_index],folds_accuracy[max_accuracy_index],folds_accuracy



print("Data loading please wait 1-2 minutes...")
Match_data = Make_Query(create_connection("database.sqlite"),"SELECT home_team_api_id, away_team_api_id, home_team_goal, away_team_goal,B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA,date FROM Match WHERE B365H != 0 AND B365D != 0 AND B365A != 0 AND BWH != 0 AND BWD != 0 AND BWA != 0 AND IWH != 0 AND IWD != 0 AND IWA != 0 AND LBH != 0 AND LBD != 0 AND LBA;","","")
Results = Data_Labels(Match_data)
vectors,output = neural_vectors(Match_data,Results)
W1,W1_BIAS,W2,W2_BIAS,ac,accuracy_array = MNN(preprocessing.scale(vectors), output)
print("The best try out of 10 folds was:")
print("The weights of layer1:", W1)
print("The bias of layer1: ", W1_BIAS)
print("The weights of layer2:", W2)
print("The bias of layer2: ", W2_BIAS)
print("The accuracy while training was: ", ac, "%")
percentage = 0
for i in range(len(accuracy_array)):
    percentage = percentage + accuracy_array[i]
percentage = percentage/len(accuracy_array)
print("The total accuracy of the NN after the 10 fold cross validation is: ",percentage,"%")