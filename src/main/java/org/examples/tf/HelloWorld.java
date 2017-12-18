package org.examples.tf;

import java.util.ArrayList;
import java.util.Comparator;

import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.examples.tf.objects.Item;
import org.jol.core.MLConf;
import org.jol.core.MLModel;
import org.jol.core.items.MLString;
import org.jol.core.MLCollection;
import org.jol.models.TFModel;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import com.fasterxml.jackson.databind.ObjectMapper;

public class HelloWorld {
  public static void main(String[] args) throws Exception {
    MLConf conf = new ObjectMapper().readValue(FileUtils.readFileToString(
        new ClassPathResource("/h_world_object.json").getFile()), MLConf.class);

//	MLString hworld = new MLString(new TFModel(conf));
    
//    System.err.println(hworld.getString());
  }
}
