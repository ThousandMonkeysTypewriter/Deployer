package org.jol.dl4j.conf;

import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.utilities.DataUtilities;
import org.deeplearning4j.utilities.SentimentExampleIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jol.core.MLConf;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class SentimentAnalyzer extends MLConf {

  private WordVectors wvs;
  private DefaultTokenizerFactory tokenizerFactory;

  public SentimentAnalyzer() {
  }
  
  public void init() throws Exception {
    wvs = WordVectorSerializer.loadStaticModel(new ClassPathResource(wordVectorsPath).getFile());

    dataPathAbsolute = new ClassPathResource(dataPathLocal).getFile().getAbsolutePath();
    modelLocationAbsolute = new ClassPathResource(modelLocationLocal).getFile().getAbsolutePath();

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    
    System.err.println(classifier);
  }
  
//  public SentimentAnalyzer (String dataPath_, String modelLocation_, String wordVectorsPath_, Boolean save_, int batchSize,
//      int vectorSize_, int nEpochs_, int truncateReviewsToLength_, String dataUrl_, String classifier_) {
//    this.dataPath = dataPath_;
//    this.modelLocation = modelLocation_;
//    this.save = save_;
//    this.classifier = classifier_;
//
//    this.batchSizeTraining = batchSize;
//    this.batchSizeTest = batchSize;
//    this.nEpochs = nEpochs_;
//    
//    System.err.println("!"+wvs);
//  }

  /**Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
   * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
   * (using the Word2Vec model) and fed into a recurrent neural network.
   * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
   * This data set contains 25,000 training reviews + 25,000 testing reviews
   *
   * Process:
   * 1. Automatic on first run of example: Download data (movie reviews) + extract
   * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
   * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
   * 4. Train network
   *
   * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
   * additional tuning.
   *
   * NOTE / INSTRUCTIONS:
   * You will have to download the Google News word vector model manually. ~1.5GB
   * The Google News vector model available here: https://code.google.com/p/word2vec/
   * Download the GoogleNews-vectors-negative300.bin.gz file
   * Then: set the WORD_VECTORS_PATH field to point to this location.
   *
   * @author Alex Black
   */
  
  public Model train (MLConf global_conf) throws Exception {
    if (wordVectorsPath.startsWith("/PATH/TO/YOUR/VECTORS/")){
      throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
    }
    
    //Download and extract data
    downloadData(global_conf);

    Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces


    //DataSetIterators for training and testing respectively
    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(wordVectorsPath));
    System.err.println(global_conf.dataPathAbsolute+", "+wordVectors+", "+batchSizeTraining+", "+truncateReviewsToLength);
    SentimentExampleIterator train = new SentimentExampleIterator(global_conf.dataPathAbsolute, wordVectors, batchSizeTraining, truncateReviewsToLength, true);
    SentimentExampleIterator test = new SentimentExampleIterator(global_conf.dataPathAbsolute, wordVectors, batchSizeTest, truncateReviewsToLength, false);

    //Set up network configuration
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .updater(Updater.ADAM)  //To configure: .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
    .regularization(true).l2(1e-5)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
    .learningRate(2e-2)
    .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
        .activation(Activation.TANH).build())
        .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
            .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .pretrain(false).backprop(true).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(1));

    System.out.println("Starting training");
    for (int i = 0; i < global_conf.nEpochs; i++) {
      net.fit(train);
      train.reset();
      System.out.println("Epoch " + i + " complete. Starting evaluation:");
    }

    //Run evaluation. This is on 25k reviews, so can take some time
    Evaluation evaluation = net.evaluate(test);
    System.out.println(evaluation.stats());

    return net;
  }

  public void downloadData(MLConf global_conf) throws Exception {
    //Create directory if required
    File directory = new File(global_conf.dataPathAbsolute);
    if(!directory.exists()) directory.mkdir();

    //Download file:
    String archizePath = global_conf.dataPathAbsolute + "aclImdb_v1.tar.gz";
    File archiveFile = new File(archizePath);
    String extractedPath = global_conf.dataPathAbsolute + "aclImdb";
    File extractedFile = new File(extractedPath);

    if( !archiveFile.exists() ){
      System.out.println("Starting data download (80MB)...");
      FileUtils.copyURLToFile(new URL(dataUrl), archiveFile);
      System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
      //Extract tar.gz file to output directory
      DataUtilities.extractTarGz(archizePath, global_conf.dataPathAbsolute);
    } else {
      //Assume if archive (.tar.gz) exists, then data has already been extracted
      System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
      if( !extractedFile.exists()){
        //Extract tar.gz file to output directory
        DataUtilities.extractTarGz(archizePath, global_conf.dataPathAbsolute);
      } else {
        System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
      }
    }
  }
  
  public INDArray prepareFeatures(String reviewContents) throws IOException, InterruptedException {
    System.err.println(wvs);
    int vectorSize = wvs.getWordVector(wvs.vocab().wordAtIndex(0)).length;

    List<String> tokens = tokenizerFactory.create(reviewContents).getTokens();
    List<String> tokensFiltered = new ArrayList<>();

    for(String t : tokens ){
      if(wvs.hasWord(t)) tokensFiltered.add(t);
    }
    int outputLength = Math.max(truncateReviewsToLength,tokensFiltered.size());

    INDArray features = Nd4j.create(1, vectorSize, outputLength);
    for( int j=0; j<tokens.size() && j<truncateReviewsToLength; j++ ){
      String token = tokens.get(j);
      INDArray vector = wvs.getWordVectorMatrix(token);
      features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
    }

    return features;
  }

  public int getIndex(INDArray output) {
    int timeSeriesLength = output.size(2);
    INDArray probabilitiesAtLastWord = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

    System.out.println("\n\nProbabilities at last time step:");
    System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
    System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

    if (probabilitiesAtLastWord.getDouble(0) > probabilitiesAtLastWord.getDouble(1))
      return 1;
    else
      return 0;
  }
}
