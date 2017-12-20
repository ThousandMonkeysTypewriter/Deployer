package org.examples.dl4j.seq2seq;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jol.core.DSL;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;



/**
 * Created by susaneraly on 3/27/16.
 */
public class AdditionRNN {

    /*
        This example is modeled off the sequence to sequence RNNs described in http://arxiv.org/abs/1410.4615
        Specifically, a sequence to sequence NN is build for the addition operation. Addition is viewed as a translation task.
        For eg. "12+23 " = " 35" with "12+23 " as the input sequence to be translated to the output sequence " 35"
        For a general idea of seq2seq models refer to the image on Pg. 3 in the paper https://arxiv.org/pdf/1406.1078v3

        This example is build using a computation graph with RNN layers.
        Refer here for more details on computation graphs in dl4j
            https://deeplearning4j.org/compgraph
        And here for RNNs
            https://deeplearning4j.org/usingrnns

        There are two RNN layers to this computation graph. The inputs to them are as follows,
        During training:
            - encoder RNN layer:
                   Takes in the addition input string, eg. '12+13'
            - decoder RNN layer:
                   Takes in an input that combines the following two elements
                      1. The output of the very last time step of the encoder
                      2. The shifted 'correct' output of the addition (by appending with a "Go"), 'Go25 '

            which is then trained to fit to the output of the decoder RNN layer, eg '25 '

        During test the inputs are as follows:
            - encoder RNN layer:
                    Takes in the addition input string '12+13'
            - decoder RNN layer:
                   For a time step t takes in an input that combines the following two elements
                      1. The output of the very last time step of the encoder
                      2. The output of the decoder at time step, t-1; For t = 0 input to the decoder is merely "go"

        One hot vectors are used for encoding/decoding (length of one hot vector is 14 for 10 digits and "+"," ",beginning of string and end of string
        10 epochs give ~85% accuracy for 2 digits
        20 epochs give >95% accuracy for 2 digits

        To try out addition for numbers with different number of digits simply change "NUM_DIGITS"
     */

    //Random number generator seed, for reproducability
    public static final int seed = 1234;

    //Tweak these to tune the dataset size = batchSize * totalBatches
    public static int batchSize = 10;
    public static int totalBatches = 500;
    public static int nEpochs = 3;
    public static int nIterations = 1;

    //Tweak the number of hidden nodes
    public static final int numHiddenNodes = 128;

    //This is the size of the one hot vector
    public static final int FEATURE_VEC_SIZE = 59;
	public static final int inputsNum = 14;
	
	static DSL dsl;

    public static void main(String[] args) throws Exception {
		
		dsl = new DSL(); 

        //DataType is set to double for higher precision
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        //This is a custom iterator that returns MultiDataSets on each call of next - More details in comments in the class
        CustomSequenceIterator iterator = new CustomSequenceIterator(seed, batchSize, totalBatches, dsl);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.25)
                .updater(Updater.ADAM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nIterations)
                .seed(seed)
                .graphBuilder()
                //These are the two inputs to the computation graph
                .addInputs("additionIn", "sumOut")
                .setInputTypes(InputType.recurrent(inputsNum), InputType.recurrent(inputsNum))
                //The inputs to the encoder will have size = minibatch x featuresize x timesteps
                //Note that the network only knows of the feature vector size. It does not know how many time steps unless it sees an instance of the data
                .addLayer("encoder", new GravesLSTM.Builder().nIn(inputsNum).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build(),"additionIn")
                //Create a vertex indicating the very last time step of the encoder layer needs to be directed to other places in the comp graph
                .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
                //Create a vertex that allows the duplication of 2d input to a 3d input
                //In this case the last time step of the encoder layer (viz. 2d) is duplicated to the length of the timeseries "sumOut" which is an input to the comp graph
                //Refer to the javadoc for more detail
                .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
                //The inputs to the decoder will have size = size of output of last timestep of encoder (numHiddenNodes) + size of the other input to the comp graph,sumOut (feature vector size)
                .addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE+numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build(), "sumOut","duplicateTimeStep")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();
//        net.setListeners(new ScoreIterationListener(1));
		
        //Train model:
        int iEpoch = 0;
        int testSize = 100;
        Seq2SeqPredicter predictor = new Seq2SeqPredicter(net);
        while (iEpoch < nEpochs) {
            net.fit(iterator);
            System.out.printf("* = * = * = * = * = * = * = * = * = ** EPOCH %d ** = * = * = * = * = * = * = * = * = * = * = * = * = * =\n",iEpoch);
            MultiDataSet testData = iterator.generateTest(testSize);
            INDArray predictions = predictor.output(testData);
            encode_decode_eval(predictions,testData.getFeatures()[0],testData.getLabels()[0]);
            /*
            (Comment/Uncomment) the following block of code to (see/or not see) how the output of the decoder is fed back into the input during test time
            
            System.out.println("Printing stepping through the decoder for a minibatch of size three:");
            testData = iterator.generateTest(3);
            predictor.output(testData,true);*/
            System.out.println("\n* = * = * = * = * = * = * = * = * = ** EPOCH " + iEpoch + " COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =");
            iEpoch++;
            ModelSerializer.writeModel(net, "/root/JOL/src/main/resources/rnn/model.zip", true);
        }

        ComputationGraph net1 = ModelSerializer.restoreComputationGraph("/root/JOL/src/main/resources/rnn/model.zip", false);
        Seq2SeqPredicter predictor1 = new Seq2SeqPredicter(net1);
        net1.fit(iterator);
        MultiDataSet testData1 = iterator.generateTest(10);
        
        INDArray predictions = predictor1.output(testData1, false);
		encode_decode_eval(predictions,testData1.getFeatures()[0],testData1.getLabels()[0]);
        
        String [] predictionS = CustomSequenceIterator.oneHotDecode(predictions);
		for (String p : predictionS) {
		  dsl.act(p.trim().replace("End", ""));
		}
        
        /*
        ComputationGraph net1 = ModelSerializer.restoreComputationGraph("/root/JOL/src/main/resources/rnn/model.zip");
        MultiDataSet testData1 = iterator.generateTest(10);
        net1.fit(iterator);
        System.err.println("!"+testData1);
//        predictor.output(testData);
        INDArray decoderInputTemplate = testData1.getFeatures()[1].dup();
        INDArray ret = net1.output(false,testData1.getFeatures()[0],decoderInputTemplate)[0];
        
        encode_decode_eval(ret,testData1.getFeatures()[0],testData1.getLabels()[0]);
        
        System.out.println("\tEncoder input and Decoder input:");
        System.out.println(CustomSequenceIterator.mapToString(testData1.getFeatures()[0],decoderInputTemplate, "="));
        System.out.println("\tDecoder output:");
        System.out.println("\t"+String.join("\n\t",CustomSequenceIterator.oneHotDecode(ret)));*/
    }

    private static void encode_decode_eval(INDArray predictions, INDArray questions, INDArray answers) {

        int nTests = predictions.size(0);
        int wrong = 0;
        int correct = 0;
        String [] questionS = CustomSequenceIterator.oneHotDecode(questions);
        String [] answersS = CustomSequenceIterator.oneHotDecode(answers);
        String [] predictionS = CustomSequenceIterator.oneHotDecode(predictions);
        for (int iTest=0; iTest < nTests; iTest++) {
            if (!answersS[iTest].equals(predictionS[iTest])) {
                System.out.println(questionS[iTest] + " gives "+ predictionS[iTest] + " != " + answersS[iTest]);
                wrong++;
            }
            else {
                System.out.println(questionS[iTest] + " gives "+ predictionS[iTest] + " == " + answersS[iTest]);
                correct++;
            }
        }
        System.out.println("WRONG: "+wrong);
        System.out.println("CORRECT: "+correct);
        System.out.println("The digits along with the spaces have to be predicted\n");
    }

}

