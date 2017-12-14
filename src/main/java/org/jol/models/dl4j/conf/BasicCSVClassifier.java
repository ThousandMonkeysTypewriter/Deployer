package org.jol.models.dl4j.conf;


import java.io.File;
import java.io.IOException;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jol.core.MLConf;
import org.jol.models.dl4j.utilities.DataUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BasicCSVClassifier extends MLConf {

  public BasicCSVClassifier() {
  }
  
//  public BasicCSVClassifier (String dataPath_, String modelLocation_, Boolean save_, int batchSizeTraining_,
//      int batchSizeTest_, int numInputs_, int numOutputs_, int nEpochs_, String classifier_) {
//    this.dataPath = dataPath_;
//    this.modelLocation = modelLocation_;
//    this.save = save_;
//    this.classifier = classifier_;
//
//    this.batchSizeTraining = batchSizeTraining_;
//    this.batchSizeTest = batchSizeTest_;
//    this.numInputs = numInputs_;
//    this.numOutputs = numOutputs_;
//    this.nEpochs = nEpochs_;
//  }

  /**
   * This example is intended to be a simple CSV classifier that seperates the training data
   * from the test data for the classification of animals. It would be suitable as a beginner's
   * example because not only does it load CSV data into the network, it also shows how to extract the
   * data and display the results of the classification, as well as a simple method to map the lables
   * from the testing data into the results.
   *
   * @author Clay Graham
   */

  public Model train (MLConf global_conf) throws Exception {
    DataSet trainingData = DataUtilities.readCSVDataset(
        new ClassPathResource("/animals/DataExamples/animals/animals_train.csv").getFile(),
        global_conf.batchSizeTraining, global_conf.numInputs, global_conf.numOutputs);


    DataSet testData = DataUtilities.readCSVDataset(new ClassPathResource("/animals/DataExamples/animals/animals.csv").getFile(),
        global_conf.batchSizeTest, global_conf.numInputs, global_conf.numOutputs);

    //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
    DataNormalization normalizer = new NormalizerStandardize();
    normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
    normalizer.transform(trainingData);     //Apply normalization to the training data
    normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

    log.info("Build model....");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .activation(Activation.TANH)
    .weightInit(WeightInit.XAVIER)
    .learningRate(0.1)
    .regularization(true).l2(1e-4)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(global_conf.numInputs).nOut(3).build())
    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
    .activation(Activation.SOFTMAX).nIn(3).nOut(global_conf.numOutputs).build())
    .backprop(true).pretrain(false)
    .build();

    //run the model
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(new ScoreIterationListener(100));

    System.out.println("Starting training");
    for (int i = 0; i < global_conf.nEpochs; i++) {
      model.fit(trainingData);
      //      trainingData.reset();
      System.out.println("Epoch " + i + " complete. Starting evaluation:");
    }

    INDArray output = model.output(testData.getFeatureMatrix());

    //evaluate the model on the test set
    Evaluation eval = new Evaluation(3);
    eval.eval(testData.getLabels(), output);

    System.out.println(eval.stats());

    return model;
  }

  public float getIndex(INDArray output) {
    return maxIndex(getFloatArrayFromSlice(output));
  }

  /**
   * This method is to show how to convert the INDArray to a float array. This is to
   * provide some more examples on how to convert INDArray to types that are more java
   * centric.
   *
   * @param rowSlice
   * @return
   */
  public static float[] getFloatArrayFromSlice(INDArray rowSlice){
    float[] result = new float[rowSlice.columns()];
    for (int i = 0; i < rowSlice.columns(); i++) {
      result[i] = rowSlice.getFloat(i);
    }
    return result;
  }

  /**
   * find the maximum item index. This is used when the data is fitted and we
   * want to determine which class to assign the test row to
   *
   * @param vals
   * @return
   */
  public static int maxIndex(float[] vals){
    int maxIndex = 0;
    for (int i = 1; i < vals.length; i++){
      float newnumber = vals[i];
      if ((newnumber > vals[maxIndex])){
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  @Override
  public INDArray prepareFeatures(String input) throws IOException,
      InterruptedException {
    // TODO Auto-generated method stub
    return null;
  }
}
