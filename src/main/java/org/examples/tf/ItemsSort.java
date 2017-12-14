package org.examples.tf;

import java.util.ArrayList;

import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.examples.tf.objects.Item;
import org.jol.core.MLConf;
import org.jol.core.MLModel;
import org.jol.models.TFModel;

import com.fasterxml.jackson.databind.ObjectMapper;

public class ItemsSort {
  public static void main(String[] args) throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();

    MLConf conf = objectMapper.readValue(FileUtils.readFileToString(
        new ClassPathResource("/animals/animals_model_conf.json").getFile()), MLConf.class);

    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    ArrayList<Item> items = new ArrayList<>();
    
    MLModel model = new TFModel(conf);
    
   /* //model inputs
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
    } */
    
    System.err.println(items);
  }

}
