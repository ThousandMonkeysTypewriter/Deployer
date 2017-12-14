package org.jol.core;
	
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;

import org.jol.core.MLItem;
	
public class MLCollection  {
	
	private ArrayList<MLItem> items;
	
	public MLCollection (ArrayList<MLItem> items_) {
		items = items_;
		//conf.prepareFeatures(ArrayList<double[]> inputDoubles)
		//model.getLabel(items);		
	}
	
	public void sort() {
		Collections.sort(items, comparator_items_labels);
	}
	  
	  public static Comparator<MLItem> comparator_items_labels = new Comparator<MLItem>() {
		public int compare(MLItem o1, MLItem o2) {
		  return o2.getLabel().compareTo(o1.getLabel());
		}
	  };
}