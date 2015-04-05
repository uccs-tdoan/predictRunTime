/* performance.java is in companion with ClassifiersCost.java (with Misclassifier Cost)
 * 
 * Tri Doan
 * Note: weka developer 3.7.2
 * Implement different classifiers in order to get statistical summaries   
 * Date: 2015, Jan 28
 * Last update: Feb 15, 2015
 * In last update, i use Stratified k fold cross validation to deal with unbalanced classes
 * instead of k-fold cross validation
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
import java.util.Random;
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

import java.lang.management.*;  // Evaluate CPU time with java
 
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

	/** Get CPU time in nanoseconds. */
	public static long getCpuTime( ) {
	    ThreadMXBean bean = ManagementFactory.getThreadMXBean( );
	    return bean.isCurrentThreadCpuTimeSupported( ) ?
	        bean.getCurrentThreadCpuTime( ) : 0L;
	}
	 
	/** Get user time in nanoseconds. */
	public static long getUserTime( ) {
	    ThreadMXBean bean = ManagementFactory.getThreadMXBean( );
	    return bean.isCurrentThreadCpuTimeSupported( ) ?
	        bean.getCurrentThreadUserTime( ) : 0L;
	}

	public long totalMem() {
        return Runtime.getRuntime().totalMemory();
    }
	
	/* ** Get system time in nanoseconds. */
	public static long getSystemTime( ) {
	    ThreadMXBean bean = ManagementFactory.getThreadMXBean( );
	    return bean.isCurrentThreadCpuTimeSupported( ) ?
	      (bean.getCurrentThreadCpuTime( ) - bean.getCurrentThreadUserTime( )) : 0L;
	}

	public static long usedMem() {
        return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
    }
	
	
	public static void main(String[] args) throws Exception {
		String outFile = "PerformanceAddLaptopTest.csv";
		String outPath = "C:\\smallProject\\predictRunTime\\results";
		
		
		String inPath  = "C:\\smallProject\\predictRunTime\\tri1";
		List<String> results = getList(inPath);
		
		String inFile;
		String outStore= outPath +"\\"+outFile;
	 
		PrintWriter out =new PrintWriter(new FileWriter(outStore,true));
		
		Runtime runtime = Runtime.getRuntime();
		
		long maxMemory = runtime.maxMemory();
        long allocatedMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        
		float avgAcc, avgRMSE,avgFscore ,avgPRC,avgAUC ,avgErr,avgCost;
		// For each training-testing split pair, train and test the classifier
	    long startTrain,startTest, totalTrain=0,totalTest=0 ;
		
	    // Remove comment to allow enter seed, fold from keyboard 
	    // int seed  = Integer.parseInt(Utils.getOption("s", args));
	    //int folds = Integer.parseInt(Utils.getOption("x", args));
	    int seed = 12;   // the seed for randomizing the data
	    int folds = 10;    // 10 cross-validation
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

 		long startSystemTimeNano =  getSystemTime( );
        long startUserTimeNano   = getUserTime( );
	
     		
     		
		for (int k=0; k< results.size();k++) {
			inFile=results.get(k);
			String inStore = inPath+"\\"+ results.get(k);
			System.out.println(inStore);
		//System.out.println(outStore);
		BufferedReader datafile = readDataFile(inPath+"\\"+inFile);
		
		//PrintWriter out = new PrintWriter(new FileWriter("c:\\AlgoSelecMeta\\output.csv"));
		
		
		Random rand = new Random(seed);   // create seeded number generator
		
		Instances data = new Instances(datafile);
		data.randomize(rand);         // randomize data with number generator
		data.setClassIndex(data.numAttributes() - 1);
	    if (data.classAttribute().isNominal())
	        data.stratify(folds);

		
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, folds);
		
		// The follow separate split into training and testing arrays in previous version 
		// Instances[] trainingSplits = split[0];
		// Instances[] testingSplits = split[1];
        // Print header
		 
		out.println("Accuracy,RMSE,Fscore,PRC,AUC,SAR,Dataset,nInstances,nClasses,nAttributes,Algorithm,runTime,CPUtime,usedMemory");
		// Run for each model
		for (int j = 0; j < models.length; j++) {
 
			
			// Collect every group of predictions for current model in a FastVector
			FastVector predictions = new FastVector();
          
            avgAcc=0; avgRMSE=0; avgFscore =0 ;avgPRC=0 ;avgAUC=0 ; avgErr=0 ;avgCost=0;
            totalTrain=0; totalTest=0;
			// For each training-testing split pair, train and test the classifier
            long taskUserTimeNano   ;
            long taskSystemTimeNano ;
           
            int count=0;
            // System.out.printf(" Number of split %d",trainingSplits.length);
			// for (int i = 0; i < trainingSplits.length; i++) {
            for (int i = 0; i < folds; i++) {
				startTrain = System.currentTimeMillis();
				// This belove of code used in previous version where data does not shuffle
				//Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);  in provisou
				// the following code is used by the StratifiedRemoveFolds filter
				Instances train = data.trainCV(folds, i);
			    Instances test = data.testCV(folds, i);
			    
				Evaluation validation = classify(models[j], train, test);
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
	               
			    /* These following code used inexplorer and experimenter of Weka
			     * Instances train = randData.trainCV(folds, n, rand);
			     *  build and evaluate classifier
      			 *	Classifier clsCopy = Classifier.makeCopy(cls);
      			 *	clsCopy.buildClassifier(train);
      			 *	eval.evaluateModel(clsCopy, test);
      			 * http://stats.stackexchange.com/questions/100631/would-a-sorted-class-harm-a-10-split-cross-validation
      			 * http://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-java/
			     * reference: http://weka.wikispaces.com/Generating+cross-validation+folds+%28Java+approach%29  
			     *  /
			     */
	            
	            
            } // End for loop - running for one model 
			
			
			
			System.out.println(" RMSE = "+String.format("%.2f", avgRMSE/folds));
			System.out.println(" Fscore = "+String.format("%.2f", avgFscore/folds));
			System.out.println(" avgPRC = "+String.format("%.2f",(float) avgPRC/folds));
			System.out.println(" avgAUC = "+String.format("%.2f",(float) avgAUC/folds));
			
			System.out.println(" run time = "+String.format("%d",(long) totalTrain+totalTest));
		    System.out.print(" Memory used"+String.format("%d",(long) usedMem()));	
			
			//System.out.println(" ratio of run time over training time = "+String.format("%f",(float)  totalTrainingTime/ duration ));
			// Calculate overall accuracy of current classifier on all splits
			//double accuracy = calculateAccuracy(predictions);
			avgAcc   = avgAcc/folds;
			avgRMSE   = avgRMSE/folds;
			avgFscore = avgFscore/folds;
			avgPRC    = avgPRC/folds;
			avgAUC    = avgAUC/folds;
			double avgSAR = (avgAcc+avgAUC+(1-avgRMSE))/3;	
			
			
			taskUserTimeNano    = getUserTime( ) - startUserTimeNano;
            taskSystemTimeNano  = getSystemTime( ) - startSystemTimeNano;
            // "User time" is the time spent running your application's own code.
            // "System time" is the time spent running OS code on behalf of your application (such as for I/O).
            // "CPU time" = user time + system time. It's the total time spent using a CPU for your application.
			out.print(String.format("%.2f",avgAcc)+","+String.format("%.2f",avgRMSE)+","+String.format("%.2f",avgFscore)+",");
			out.print(String.format("%.2f",avgPRC)+","+String.format("%.2f",avgAUC)+",");
			out.print(String.format("%.2f",avgSAR)+","+ inFile.split("\\.")[0]+","+String.format("%d", data.numInstances())+",");
			out.print(String.format("%d",data.numClasses())+","+String.format("%d", data.numAttributes()-1)+",");
		    out.print(models[j].getClass().getSimpleName()+","+String.format("%d",totalTrain)+",");
		    out.println(String.format("%d",(taskUserTimeNano+taskSystemTimeNano)/1000000000));
		    out.println(String.format("%d",usedMem()/1024/1024));
	/*
	 *  Alternatively, to compute memory usage in Megabytes
	 * double currentMemory = ( (double)((double)(Runtime.getRuntime().totalMemory()/1024)/1024))- 
	 *     ((double)((double)(Runtime.getRuntime().freeMemory()/1024)/1024));
	 * 	    
	 */
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

