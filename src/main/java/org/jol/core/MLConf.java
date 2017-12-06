package org.jol.core;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.api.Model;
import org.jol.dl4j.conf.BasicCSVClassifier;
import org.jol.dl4j.conf.CSVClassifier;
import org.jol.dl4j.conf.ImageClassifier;
import org.jol.dl4j.conf.SentimentAnalyzer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import org.jol.dl4j.conf.BasicCSVClassifier;

/**
 * Returns the data which is necessary for loading and using Model. Deserializes a data from JSON.
 * Needs JsonSubTypes - type of configuration and model that will be loaded
 * 
 * @author  Fedor Soprunov
 * @see     BasicCSVClassifier
 */

@JsonIgnoreProperties(ignoreUnknown = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY)
@JsonSubTypes({
  @JsonSubTypes.Type(value = BasicCSVClassifier.class, name = "BasicCSVClassifier"),
  @JsonSubTypes.Type(value = CSVClassifier.class, name = "CSVClassifier"),
  @JsonSubTypes.Type(value = ImageClassifier.class, name = "ImageClassifier"),
  @JsonSubTypes.Type(value = SentimentAnalyzer.class, name = "SentimentAnalyzer") }
    )

public abstract class MLConf {

  public boolean create = false;
  /**
   * Path to the model storage
   */
  public String dataPathLocal, modelLocationLocal, dataPathAbsolute, modelLocationAbsolute;
  /**
   * Possible output labels
   */
  public String classifier;

  /**
   * Conf for training common
   */
  public static Logger log = LoggerFactory.getLogger(MLConf.class);
  public int nEpochs;

  /**
   * 5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
   */
  public int numInputs; 
  /**
   * 3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2
   */
  public int numOutputs; 
  /**
   * Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
   */
  public int batchSizeTraining;    
  public int batchSizeTest;

  public int seed;
  public int iterations;
  public double testTrainSplit;

  /**
   * Conf for training datavec
   */
  public String wordVectorsPath;
  public int vectorSize;
  public int truncateReviewsToLength;
  public String dataUrl;

  /**
   * Conf for image classifier
   */

  public int height;
  public int width;
  public int channels;
  public int numExamples;

  public int listenerFreq;
  public double splitTrainTest;

  public String modelType; 

  /**
   * JACKSON polymorphism needs only empty constructor(???)
   * @throws Exception 
   */
  public void init() throws Exception {
    if ( dataPathLocal != null )
      dataPathAbsolute = new ClassPathResource(dataPathLocal).getFile().getAbsolutePath();
    
    modelLocationAbsolute = new ClassPathResource(modelLocationLocal).getFile().getAbsolutePath();
    System.err.println("#");
  }

  /**
   * Create trained model
   */
  public abstract Model train (MLConf global_conf) throws Exception;

  /**
   * Get label num
   */
  public abstract int getIndex (INDArray output);

  /**
   * Featurize string
   */
  public abstract INDArray prepareFeatures(String input) throws IOException, InterruptedException;
}



