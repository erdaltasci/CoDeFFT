function  ClassifyKNNCifar10
    clear all
   %load features and labels
    load setCifar10
    load cifar10TLvgg19mb32f64.mat
    P1 = P64;
    P = Ptest64;
    load cifar10TLalexnetmb32f64.mat
    load cifar10DLtrainresnet101pool5.mat
    load cifar10DLtestresnet101pool5.mat
    
    k=1; % k value of k-nn
    
    %combine training features
    P1 = [double(features); double(P1);double(P64)];
    T = trainingSet.Labels;
    classifier  = fitcknn(P1', T, 'NumNeighbors', k);
    %combine test features
    Ptest = [double(testFeatures)' double(P)' double(Ptest64)'];

    % Pass CNN image features to trained classifier
    predictedLabels = predict(classifier, Ptest);
    testLabels = testSet.Labels;
    predictedLabels = categorical(predictedLabels);

    disp(strcat('K-NN:',num2str(k),','));
    acc = sum(testLabels== predictedLabels)/length(testLabels)*100;
    disp(strcat('Acc: %',num2str(acc)));

end