package org.jol;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.classifier.animals.Animal;
import org.deeplearning4j.classifier.animals.Flower;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.utilities.DataUtilities;
import org.jol.objects.MLConf;
import org.jol.objects.MLModel;
import org.jol.objects.model.DL4JModel;
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

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * @author Adam Gibson
 */
public class IrisClassifier {

  public static void main(String[] args) throws  Exception {
    ObjectMapper objectMapper = new ObjectMapper();

    MLConf conf = objectMapper.readValue(FileUtils.readFileToString(
        new File("/root/JOL/src/main/resources/flowers/iris_model_conf.json")), MLConf.class);

    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    Map<String, ArrayList<Flower>> flowers = new HashMap<String,ArrayList<Flower>>();

    //model inputs
    DataSet testData = DataUtilities.readCSVDataset(new ClassPathResource("/flowers/iris.txt").getFile(),
        conf.batchSizeTest, conf.numInputs, conf.numOutputs);

    //labels for MLItems objects
    Map<Integer,String[]> data = DataUtilities.readEnumCSV(new ClassPathResource("/flowers/iris_data.txt").getFile());

    MLModel model = new DL4JModel(conf);

    INDArray features = model.prepareFeatures(testData, true);

    for (int i = 0; i < features.rows() ; i++) {
      INDArray slice = features.slice(i);

      Flower iris = new Flower(slice, model, data.get(i));

      String label = iris.getLabel();
      if (!flowers.containsKey(label))
        flowers.put(label, new ArrayList<>());

      flowers.get(label).add(iris);
    }

    System.err.println(flowers);
  }
}