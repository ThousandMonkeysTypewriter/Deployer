package org.jol.objects;

import java.io.IOException;

import org.deeplearning4j.nn.api.Model;
import org.jol.objects.conf.BasicCSVClassifier;
import org.jol.objects.conf.CSVClassifier;
import org.jol.objects.conf.SentimentAnalyzer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY)
@JsonSubTypes({
  @JsonSubTypes.Type(value = BasicCSVClassifier.class, name = "BasicCSVClassifier"),
  @JsonSubTypes.Type(value = CSVClassifier.class, name = "CSVClassifier"),
  @JsonSubTypes.Type(value = SentimentAnalyzer.class, name = "SentimentAnalyzer") }
)

public abstract class MLConf {

  /**
   * conf for restoring
   */
  public boolean create = false;
  public String dataPath;
  public boolean save = false;
  public String modelLocation;
  public String classifier;
  
  /**
   * conf for training common
   */
  public static Logger log = LoggerFactory.getLogger(MLConf.class);
  public int nEpochs;
  public int numInputs; //5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
  public int numOutputs; //3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2

  public int batchSizeTraining;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
  //this is the data we want to classify
  public int batchSizeTest;
  
  public int seed;
  public int iterations;
  public double testTrainSplit;
  
  /**
   * conf for training datavec
   */
  public String wordVectorsPath;
  public int vectorSize;
  public int truncateReviewsToLength;
  public String dataUrl;
  
  /**
   * JACKSON polymorphism needs only empty constructor(???)
   */
  public abstract void init();

  public abstract Model train (MLConf global_conf) throws Exception;
  
  public abstract int getIndex (INDArray output);

  public abstract INDArray prepareFeatures(String input) throws IOException, InterruptedException;
}



