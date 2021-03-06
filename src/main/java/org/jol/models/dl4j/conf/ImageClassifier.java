package org.jol.models.dl4j.conf;

import org.jol.core.MLConf;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ImageClassifier extends MLConf {

  protected Random rng = new Random(seed);

  public Model train (MLConf global_conf) throws Exception {

    log.info("Load data....");
    /**cd
     * Data Setup -> organize and limit data file paths:
     *  - mainPath = path to image files
     *  - fileSplit = define basic dataset split with limits on format
     *  - pathFilter = define additional file load filter to limit size and balance batch content
     **/
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    System.out.println(System.getProperty("user.dir") + "======" + "src/main/resources/images/examples/");
    File mainPath = new File(System.getProperty("user.dir"), "src/main/resources/images/examples/");
    FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
    BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numOutputs, batchSizeTest);

    /**
     * Data Setup -> train test split
     *  - inputSplit = define train and test split
     **/
    InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
    InputSplit trainData = inputSplit[0];
    InputSplit testData = inputSplit[1];

    /**
     * Data Setup -> transformation
     *  - Transform = how to tranform images and generate large dataset to train on
     **/
    ImageTransform flipTransform1 = new FlipImageTransform(rng);
    ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
    ImageTransform warpTransform = new WarpImageTransform(rng, 42);
    //        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
    List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});

    /**
     * Data Setup -> normalization
     *  - how to normalize images and generate large dataset to train on
     **/
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

    log.info("Build model....");

    // Uncomment below to try AlexNet. Note change height and width to at least 100
    //        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

    MultiLayerNetwork network;
    switch (modelType) {
    case "LeNet":
      network = lenetModel();
      break;
    case "AlexNet":
      network = alexnetModel();
      break;
    case "custom":
      network = customModel();
      break;
    default:
      throw new InvalidInputTypeException("Incorrect model provided.");
    }
    network.init();
    // network.setListeners(new ScoreIterationListener(listenerFreq));
    UIServer uiServer = UIServer.getInstance();
    StatsStorage statsStorage = new InMemoryStatsStorage();
    uiServer.attach(statsStorage);
    network.setListeners((IterationListener)new StatsListener( statsStorage),new ScoreIterationListener(iterations));
    /**
     * Data Setup -> define how to load data into net:
     *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
     *  - dataIter = a generator that only loads one batch at a time into memory to save memory
     *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
     **/
    ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
    DataSetIterator dataIter;
    MultipleEpochsIterator trainIter;


    log.info("Train model....");
    // Train without transformations
    recordReader.initialize(trainData, null);
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSizeTest, 1, numOutputs);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);
    trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
    network.fit(trainIter);

    // Train with transformations
    for (ImageTransform transform : transforms) {
      System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
      recordReader.initialize(trainData, transform);
      dataIter = new RecordReaderDataSetIterator(recordReader, batchSizeTest, 1, numOutputs);
      scaler.fit(dataIter);
      dataIter.setPreProcessor(scaler);
      trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
      network.fit(trainIter);
    }

    log.info("Evaluate model....");
    recordReader.initialize(testData);
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSizeTest, 1, numOutputs);
    scaler.fit(dataIter);
    dataIter.setPreProcessor(scaler);
    Evaluation eval = network.evaluate(dataIter);
    log.info(eval.stats(true));

    return network;
  }

  private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
    return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
  }

  private ConvolutionLayer conv3x3(String name, int out, double bias) {
    return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
  }

  private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
    return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
  }

  private SubsamplingLayer maxPool(String name,  int[] kernel) {
    return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
  }

  private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
    return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
  }

  public MultiLayerNetwork lenetModel() {
    /**
     * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
     * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
     **/
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .regularization(false).l2(0.005) // tried 0.0001, 0.0005
    .activation(Activation.RELU)
    .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
    .weightInit(WeightInit.XAVIER)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(new Nesterovs(0.9))
    .list()
    .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
    .layer(1, maxPool("maxpool1", new int[]{2,2}))
    .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
    .layer(3, maxPool("maxool2", new int[]{2,2}))
    .layer(4, new DenseLayer.Builder().nOut(500).build())
    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
    .nOut(numOutputs)
    .activation(Activation.SOFTMAX)
    .build())
    .backprop(true).pretrain(false)
    .setInputType(InputType.convolutional(height, width, channels))
    .build();

    return new MultiLayerNetwork(conf);

  }

  public MultiLayerNetwork alexnetModel() {
    /**
     * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
     * and the imagenetExample code referenced.
     * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
     **/

    double nonZeroBias = 1;
    double dropOut = 0.5;

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .weightInit(WeightInit.DISTRIBUTION)
    .dist(new NormalDistribution(0.0, 0.01))
    .activation(Activation.RELU)
    .updater(new Nesterovs(0.9))
    .iterations(iterations)
    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(1e-2)
    .biasLearningRate(1e-2*2)
    .learningRateDecayPolicy(LearningRatePolicy.Step)
    .lrPolicyDecayRate(0.1)
    .lrPolicySteps(100000)
    .regularization(true)
    .l2(5 * 1e-4)
    .list()
    .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
    .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
    .layer(2, maxPool("maxpool1", new int[]{3,3}))
    .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
    .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
    .layer(5, maxPool("maxpool2", new int[]{3,3}))
    .layer(6,conv3x3("cnn3", 384, 0))
    .layer(7,conv3x3("cnn4", 384, nonZeroBias))
    .layer(8,conv3x3("cnn5", 256, nonZeroBias))
    .layer(9, maxPool("maxpool3", new int[]{3,3}))
    .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
    .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
    .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
    .name("output")
    .nOut(numOutputs)
    .activation(Activation.SOFTMAX)
    .build())
    .backprop(true)
    .pretrain(false)
    .setInputType(InputType.convolutional(height, width, channels))
    .build();

    return new MultiLayerNetwork(conf);

  }

  public static MultiLayerNetwork customModel() {
    /**
     * Use this method to build your own custom model.
     **/
    return null;
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
