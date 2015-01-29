/* performance.java is in companion with ClassifiersCost.java (with Misclassifier Cost)
 * 
 * Tri Doan
 * Note: weka developer 3.7.2
 * Implement different classifiers in order to get statistical summaries   
 * Date: 2015, Jan 28
 * */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.*;
import weka.classifiers.lazy.*;
import weka.classifiers.functions.*;
import weka.classifiers.meta.*;
//import weka.classifiers.mi.*;
import weka.classifiers.bayes.*;
import weka.classifiers.trees.*;
import libsvm.*;

import weka.core.FastVector;
import weka.core.Instances;

 
public class performances {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static Evaluation classify(Classifier model,
			Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
 
	public static List<String> getList(String path) {
		   List<String> results = new ArrayList<String>();

	          File[] files = new File(path).listFiles();
	         //If this pathname does not denote a directory, then listFiles() returns null. 
	    
	         for (File file : files) {
	             if (file.isFile() && file.getName().endsWith((".arff"))) {
	                   results.add(file.getName());
	                    }
	                }
		   return results;
	   }

	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static void main(String[] args) throws Exception {
		String outFile = "PerformanceNoCost1.csv";
		String outPath = "C:\\data\\runtime";
		
		
		// For test purpose only
		
		//List<String> results = new ArrayList<String>() ;
		//List<String> results = Arrays.asList("zoo.arff","vehicle.arff");
		
		//String inFile  = "zoo.arff";
//		String inPath  = "c:\\data\\uci";
		String inPath  = "C:\\data\\tmp";
		List<String> results = getList(inPath);
		
		String inFile;
		String outStore= outPath +"\\"+outFile;
	 
		PrintWriter out =new PrintWriter(new FileWriter(outStore,true));
		
		float avgAcc, avgRMSE,avgFscore ,avgPRC,avgAUC ,avgErr,avgCost;
		// For each training-testing split pair, train and test the classifier
	    long startTrain,startTest, totalTrain=0,totalTest=0 ;
		
        
     // Use a set of classifiers
     		Classifier[] models = { 
     				
     				new DecisionStump(), //one-level decision tree
     			//	new MLPClassifier(),
     				
     				
     				new LibSVM(),
     				new MultilayerPerceptron(),
     		//		new MultilayerPerceptronCS(),
     				new Logistic(),
     				new SimpleLogistic(), // linear logistic regression models. 
     				new SMO(),
     				
     			 //	new EnsembleSelection(),
                  //  new CAAR(),   
     				new IBk(), // instance based classifier used K nearest neighbor
     				new KStar(),  // instance based classifier using entropy based distance 
     				new LWL(), // Locally weighted learning used KNN
     				
     				new BayesNet(),
     				new NaiveBayes(),
     				new NaiveBayesUpdateable(),
     				new AdaBoostM1(),
     				new Bagging(),
     			
     				new Stacking(),
     			//	new StackingC(),
     				new LogitBoost(),
     			//	new MultiBoostAB(),
     				new Vote(),
     				
     				new DecisionTable(),//decision table majority classifier
     			//	new DNTB(),
     				new JRip(),
     				new OneR(),
     			//	new Ridor(),
     				new ZeroR(),
     				new PART(),
     				
     				
     				
     				new RandomCommittee(),
     		//		new DTNB(),
     				new J48(), // a decision trees
     		//		new BFTree(),
     		//		new FT(),
     		//		new LADTree(),
     				new LMT(),
     		//		new NBTree(),
     				new RandomForest(),
     				new RandomTree(),
     				new REPTree(),
     			//	new SimpleCart(),
     				 
     				
     		};

		for (int k=0; k< results.size();k++) {
			inFile=results.get(k);
			String inStore = inPath+"\\"+ results.get(k);
			System.out.println(inStore);
		//System.out.println(outStore);
		BufferedReader datafile = readDataFile(inPath+"\\"+inFile);
		
		//PrintWriter out = new PrintWriter(new FileWriter("c:\\AlgoSelecMeta\\output.csv"));
		
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);

		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, 10);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		 
        // Print header
		 
		out.println("Accuracy,RMSE,Fscore,PRC,AUC,SAR,Dataset,nInstances,nClasses,nAttributes,Algorithm,trainTime,testTime");
		// Run for each model
		for (int j = 0; j < models.length; j++) {
 
			
			// Collect every group of predictions for current model in a FastVector
			FastVector predictions = new FastVector();
          
            avgAcc=0; avgRMSE=0; avgFscore =0 ;avgPRC=0 ;avgAUC=0 ; avgErr=0 ;avgCost=0;
            totalTrain=0; totalTest=0;
			// For each training-testing split pair, train and test the classifier
           
            int count=0;
            System.out.printf(" Number of split %d",trainingSplits.length);
			for (int i = 0; i < trainingSplits.length; i++) {
				startTrain = System.currentTimeMillis();
				Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
				totalTrain = totalTrain + System.currentTimeMillis()- startTrain;
				startTest  = System.currentTimeMillis() ;
				predictions.appendElements(validation.predictions());
				totalTest  = totalTest+ System.currentTimeMillis() - startTest;
				 avgAcc += validation.pctCorrect();
					
	             avgRMSE   += validation.rootMeanSquaredError();
	           
	             avgErr    += validation.errorRate(); 
	             avgCost   += validation.avgCost(); //  total cost of misclassifications (incorrect plus unclassified) over the total number of instances.
	                           
	             avgPRC += validation.weightedAreaUnderPRC();
	             avgAUC += validation.weightedAreaUnderROC();
	             avgFscore += validation.weightedFMeasure();
	               
			} // End for loop - running for one model 
			
			
			
			System.out.println(" RMSE = "+String.format("%.2f", avgRMSE/trainingSplits.length));
			System.out.println(" Fscore = "+String.format("%.2f", avgFscore/trainingSplits.length));
			System.out.println(" avgPRC = "+String.format("%.2f",(float) avgPRC/trainingSplits.length));
			System.out.println(" avgAUC = "+String.format("%.2f",(float) avgAUC/trainingSplits.length));
			
			System.out.println(" trainning time = "+String.format("%d",(long) totalTrain));
			System.out.println(" test time = "+String.format("%d",(long) totalTest));
			
			//System.out.println(" ratio of run time over training time = "+String.format("%f",(float)  totalTrainingTime/ duration ));
			// Calculate overall accuracy of current classifier on all splits
			//double accuracy = calculateAccuracy(predictions);
			avgAcc   = avgAcc/trainingSplits.length;
			avgRMSE   = avgRMSE/trainingSplits.length;
			avgFscore = avgFscore/trainingSplits.length;
			avgPRC    = avgPRC/trainingSplits.length;
			avgAUC    = avgAUC/trainingSplits.length;
			double avgSAR = (avgAcc+avgAUC+(1-avgRMSE))/3;	
			
			out.print(String.format("%.2f",avgAcc)+","+String.format("%.2f",avgRMSE)+","+String.format("%.2f",avgFscore)+",");
			out.print(String.format("%.2f",avgPRC)+","+String.format("%.2f",avgAUC)+",");
			out.print(String.format("%.2f",avgSAR)+","+ inFile.split("\\.")[0]+","+String.format("%d", data.numInstances())+",");
			out.print(String.format("%d",data.numClasses())+","+String.format("%d", data.numAttributes()-1)+",");
		    out.print(models[j].getClass().getSimpleName()+","+String.format("%d",totalTrain)+",");
		    out.println(String.format("%d",totalTest));
			
		    // Print current classifier's name and accuracy in a complicated,
			
			System.out.println(j);
			System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", avgAcc)
            	+ "\n---------------------------------");
			
		}  // End for loop - running for all models 
		
	
		System.out.println(" Number of Classes = "+String.format("%d",data.numClasses()));
		System.out.printf(" Number of Attributes = "+String.format("%d", data.numAttributes()-1));
		
		}
		out.close();
	}
}

