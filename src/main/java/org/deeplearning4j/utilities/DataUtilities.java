package org.deeplearning4j.utilities;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 27/01/2017.
 */
public class DataUtilities {

  private static final int BUFFER_SIZE = 4096;

  public static void extractTarGz(String filePath, String outputPath) throws IOException {
    int fileCount = 0;
    int dirCount = 0;
    System.out.print("Extracting files");
    try(TarArchiveInputStream tais = new TarArchiveInputStream(
        new GzipCompressorInputStream( new BufferedInputStream( new FileInputStream(filePath))))){
      TarArchiveEntry entry;

      /** Read the tar entries using the getNextEntry method **/
      while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
        //System.out.println("Extracting file: " + entry.getName());

        //Create directories as required
        if (entry.isDirectory()) {
          new File(outputPath + entry.getName()).mkdirs();
          dirCount++;
        }else {
          int count;
          byte data[] = new byte[BUFFER_SIZE];

          FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
          BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
          while ((count = tais.read(data, 0, BUFFER_SIZE)) != -1) {
            dest.write(data, 0, count);
          }
          dest.close();
          fileCount++;
        }
        if(fileCount % 1000 == 0) System.out.print(".");
      }
    }

    System.out.println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath);
  }

  /**
   * used for testing and training
   *
   * @param csvFileClasspath
   * @param batchSize
   * @param labelIndex
   * @param numClasses
   * @return
   * @throws IOException
   * @throws InterruptedException
   */
  public static DataSet readCSVDataset(
      File csvFile, int batchSize, int labelIndex, int numClasses)
          throws IOException, InterruptedException{

    RecordReader rr = new CSVRecordReader();
    rr.initialize(new FileSplit(csvFile));
    DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
    return iterator.next();
  }

  public static Map<Integer, String[]> readEnumCSV(File csvFile) {
    try{
      List<String> lines = IOUtils.readLines(new FileInputStream(csvFile));
      Map<Integer, String[]> enums = new HashMap<Integer, String[]>();
	  
	  int count = 0;
	  
      for(String line:lines){
        String[] parts = line.split(",");
        enums.put(count, parts);
		count++;
      }
	  
      return enums;
    } catch (Exception e){
      e.printStackTrace();
      return null;
    }

  }
}

