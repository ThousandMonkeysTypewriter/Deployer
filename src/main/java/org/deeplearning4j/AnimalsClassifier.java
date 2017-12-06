package org.deeplearning4j;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.objects.Animal;
import org.deeplearning4j.utilities.DataUtilities;
import org.jol.core.MLConf;
import org.jol.core.MLModel;
import org.jol.dl4j.model.DL4JModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import com.fasterxml.jackson.databind.ObjectMapper;

public class AnimalsClassifier {

  public static void main(String[] args) throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();

    MLConf conf = objectMapper.readValue(FileUtils.readFileToString(
        new ClassPathResource("/animals/animals_model_conf.json").getFile()), MLConf.class);

    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    Map<String, ArrayList<Animal>> animals = new HashMap<String,ArrayList<Animal>>();
    
    //model inputs
    DataSet testData = DataUtilities.readCSVDataset(new ClassPathResource("/animals/DataExamples/animals/animals.csv").getFile(),
        conf.batchSizeTest, conf.numInputs, conf.numOutputs);
		
    //labels for MLItems objects
	Map<Integer,String[]> data = DataUtilities.readEnumCSV(new ClassPathResource("/animals/DataExamples/animals/animals_labels.csv").getFile());

    MLModel model = new DL4JModel(conf);

    INDArray features = model.prepareFeatures(testData, true);

    for (int i = 0; i < features.rows() ; i++) {
      INDArray slice = features.slice(i);

      Animal animal = new Animal(slice, model, data.get(i));
      
      String label = animal.getLabel();
      if (!animals.containsKey(label))
        animals.put(label, new ArrayList<>());

      animals.get(label).add(animal);
    }
    
    System.err.println(animals);
  }
}
