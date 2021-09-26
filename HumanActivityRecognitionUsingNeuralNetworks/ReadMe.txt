Final Paper:	CT2_FinalPaper__HAR&NN

Final Code:

In DataProcess folder:
	A.there are three Programs:
		1. Preprocess_Tool.py is an utility program for Filtered_Process
		2. RawDataPreprocess.py is to deal with raw data (Including Segmentation, Extraction), in the end, it would combine all experiments into one input file for classifier
		3. Anova_Ftest.py is to fo feature selection, in the end, it would create the table to with relevant features

	B. RawData: It's raw data from "HAPT Data Set" originally, but inside it, the data are separated into: ACC and Gyro (represents acceleration and angular), and on label data
		(Inside ACC and Gyro , there no files, please put raw file from "HAPT Data Set" to there and seperate to ACC and Gyro)
		 FilteredData: It stores the result from FIltered_process.py
	C.Table: it store data after execute "RawDataPreprocess"	
	
	D.Draw_filter: Inside it, there is a program. It's to visualize one example in filter process, and separate it into body part and gravity part

In MLPClassifier:
	A. There are four python programs
		1. Classifier: The main program which loads the dataset, builds the model, trains the model and gives the metrics of the MLP model
		2. DeclareConstants: it is used for declaring the constants like paths
		3. Plots: It plots the confusion matrix and accuaracy-loss plots for train and validation datasets		
		4. Utils: It contains utility functions for small repetitive functions
	B. There are two folders
		1. Dataset: it contains the extracted features dataset
		2. Logs: It is used to store the logs of classifier and the plots of confusion-matrix and accuracy-loss
	