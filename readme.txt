1. Folder description

(1) Best models: used to store models with good test results.

(2) Data: store training and test data. The train under this folder stores training data, and the test stores test data.

(3) Models: store models

(4) PretrainedModels: used to store pre trained models on the CK+data set.

2. Procedure description


£¨1£©multi_classifer_v1.py is used to obtain the performance indicators of the two models, including accuracy, AUC, sensitivity, specificity, etc.


£¨2£©multi_ classifer_ V2. py realizes two voting situations of the multiple classifiers with two participating models, and outputs the results.


£¨3£©multi_ classifer_ V3.py realizes three voting situations of the multiple classifiers with three participating models, and outputs the results.



£¨4£©multi_ classifer_ V4. py realizes foure voting situations of the multiple classifiers with four participating models, and outputs the results.



Note: The procedure of two-stage migration learning (training on CK+dataset, and then migrating to autistic children's facial image dataset training) can be found at: https://github.com/guangxinnliying/AutismDetection.