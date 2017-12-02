package org.jol;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.classifier.reviews.Review;
import org.jol.objects.MLConf;
import org.jol.objects.MLItem;
import org.jol.objects.MLModel;
import org.jol.objects.model.DL4JModel;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

public class ReviewsFeeder {

  public static void main(String[] args) throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();

    MLConf conf = objectMapper.readValue(FileUtils.readFileToString(
        new File("/root/JOL/src/main/resources/review/sentiment_model_conf.json")), MLConf.class);
    
    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    MLModel model = new DL4JModel(conf);

    File[] files = new File(conf.dataPath+"aclImdb/test/neg/").listFiles();

    String text = FileUtils.readFileToString(files[1]);
	
    Review review = new Review(model.prepareFeatures(text), model, text);
	
    System.err.println(review);
  }
}
