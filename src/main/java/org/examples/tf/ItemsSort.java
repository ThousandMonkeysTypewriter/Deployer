package org.examples.tf;

import java.util.ArrayList;
import java.util.Comparator;

import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.examples.tf.objects.Item;
import org.jol.core.MLConf;
import org.jol.core.MLModel;
import org.jol.core.MLItem;
import org.jol.core.MLCollection;
import org.jol.models.TFModel;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import com.fasterxml.jackson.databind.ObjectMapper;

public class ItemsSort {
  public static void main(String[] args) throws Exception {
    ObjectMapper objectMapper = new ObjectMapper();

    MLConf conf = objectMapper.readValue(FileUtils.readFileToString(
        new ClassPathResource("/animals/animals_model_conf.json").getFile()), MLConf.class);

    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    ArrayList<MLItem> items = new ArrayList<>();
    
    MLModel model = new TFModel(conf);
    
	MLCollection list = new MLCollection(items);
	
	list.sort();
    
    System.err.println(items);
  }
}
