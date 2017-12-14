package org.examples.dl4j;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.examples.dl4j.objects.Animal;
import org.examples.dl4j.objects.Image;
import org.examples.dl4j.objects.Review;
import org.jol.core.MLConf;
import org.jol.core.MLModel;
import org.jol.dl4j.utilities.DataBind;
import org.jol.dl4j.utilities.DataUtilities;
import org.jol.models.DL4JModel;
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
        new ClassPathResource("/images/images_model_conf.json").getFile()), MLConf.class);

    if (args.length > 0 && args[0].equals("create")) 
      conf.create = true;

    MLModel model = new DL4JModel(conf);
    
    //model inputs and data
    DataBind testDataBind = DataUtilities.readImageFiles(new File(System.getProperty("user.dir"), "src/main/resources/images/examples/"),
        conf);

    Image img = new Image(testDataBind.getDataSet().getFeatures(), model, testDataBind.getPath(0));
    System.out.print(img);
  }
}
