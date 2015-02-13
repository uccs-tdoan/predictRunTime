/* Stack.java: implementatio for stack method
 * 
 * Tri Doan
 * Date: Feb 12, 2015
 * */
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;    
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Stacking;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.classifiers.*;

public class stack {

 public static void main(String[] args) throws Exception{ 
  BufferedReader breader=new BufferedReader(new FileReader("C:/data/big/pca_krvskp.arff")); 
  Instances train=new Instances(breader); 
  train.setClassIndex(train.numAttributes()-1); 
  breader.close(); 

  Stacking nb=new Stacking(); 
  J48 j48=new J48(); 
  SMO smo=new SMO(); 
 
    
  Classifier[] stackoptions = new Classifier[1]; 
  {
    stackoptions[0] = smo;
        }
  nb.setClassifiers(stackoptions);


nb.setMetaClassifier(j48); 
nb.buildClassifier(train); 
Evaluation eval=new Evaluation(train); 
eval.crossValidateModel(nb, train, 10, new Random(1)); 
System.out.println(eval.toSummaryString("results",true)); 
                        }} 