package org.jol.dl4j.conf;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class CSVClassifier extends MLConf {

  public CSVClassifier() {
  }

  public Model train (MLConf global_conf) throws Exception {

    //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
    int numLinesToSkip = 0;
    char delimiter = ',';
    RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
    recordReader.initialize(new FileSplit(new File(dataPathAbsolute)));

    DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSizeTraining,numInputs,numOutputs);
    DataSet allData = iterator.next();
    allData.shuffle();
    SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(testTrainSplit);  //Use 65% of data for training

    DataSet trainingData = testAndTrain.getTrain();
    DataSet testData = testAndTrain.getTest();

    //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
    DataNormalization normalizer = new NormalizerStandardize();
    normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
    normalizer.transform(trainingData);     //Apply normalization to the training data
    normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .activation(Activation.TANH)
    .weightInit(WeightInit.XAVIER)
    .learningRate(0.1)
    .regularization(true).l2(1e-4)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
        .build())
        .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
            .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(3).nOut(numOutputs).build())
            .backprop(true).pretrain(false)
            .build();

    //run the model
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(new ScoreIterationListener(100));

    model.fit(trainingData);

    //evaluate the model on the test set
    Evaluation eval = new Evaluation(3);
    INDArray output = model.output(testData.getFeatureMatrix());
    eval.eval(testData.getLabels(), output);

    return model;
  }

  public INDArray prepareFeatures(String input) throws IOException,
  InterruptedException {
    // TODO Auto-generated method stub
    return null;
  }

  public int getIndex(INDArray output) {
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
}