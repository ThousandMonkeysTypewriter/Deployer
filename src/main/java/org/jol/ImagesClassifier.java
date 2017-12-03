package org.jol;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.classifier.animals.Animal;
import org.deeplearning4j.classifier.reviews.Review;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.utilities.DataUtilities;
import org.jol.objects.MLConf;
import org.jol.objects.MLModel;
import org.jol.objects.model.DL4JModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImagesClassifier {

  public static void main(String[] args) throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();

    MLConf conf = objectMapper.readValue(FileUtils.readFileToString(
        new File("/root/JOL/src/main/resources/images/images_model_conf.json")), MLConf.class);

    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    Map<String, ArrayList<Animal>> animals = new HashMap<String,ArrayList<Animal>>();
    
    //model inputs
    DataSet testData = DataUtilities.readImageFiles(new File(System.getProperty("user.dir"), "src/main/resources/animals/pics/"),
        conf);
        
    //labels for MLItems objects
    Map<Integer,String[]> data = DataUtilities.readEnumCSV(new ClassPathResource("/animals/DataExamples/animals/animals_labels.csv").getFile());

    MLModel model = new DL4JModel(conf);

//    INDArray features = model.prepareFeatures(testData, true);
//
//    for (int i = 0; i < features.rows() ; i++) {
//      INDArray slice = features.slice(i);
//
//      Animal animal = new Animal(slice, model, data.get(i));
//      
//      String label = animal.getLabel();
//      if (!animals.containsKey(label))
//        animals.put(label, new ArrayList<>());
//
//      animals.get(label).add(animal);
//    } 
//    
//    System.err.println(animals);
    Review review = new Review(testData.getFeatures(), model, "");
    System.out.print(review);
  }
}
