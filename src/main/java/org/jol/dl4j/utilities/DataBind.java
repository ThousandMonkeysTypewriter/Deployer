package org.jol.dl4j.utilities;

import java.net.URI;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

public class DataBind {

  private RecordReaderDataSetIterator dataIter;
  private URI[] locations;

  public DataBind(RecordReaderDataSetIterator dataIter_, URI[] locations_) {
     dataIter = dataIter_;
     locations = locations_;
  }
  
  public DataSet getDataSet () {
    dataIter.reset();
    return dataIter.next();
  }

  public String getPath(int i) {
    return locations[i].getPath();
  }
}
