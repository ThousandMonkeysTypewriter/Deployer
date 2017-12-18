package org.jol.core;

import java.util.HashMap;
import java.util.Map;

public class DSL {
  private static final Map<Integer, String> codesToCommands = new HashMap<Integer, String>();
  private static final Map<String, Integer> commandsToCodes = new HashMap<String, Integer>();

  public DSL() {
    
    int count = 0;
    
    for(char c = 'A'; c <= 'Z'; ++c) {
      codesToCommands.put(count, String.valueOf(c));
      commandsToCodes.put(String.valueOf(c), count);
      count++;
    }
    
    for(char c = 'a'; c <= 'z'; ++c) {
      codesToCommands.put(count, String.valueOf(c));
      commandsToCodes.put(String.valueOf(c), count);
      count++;
    }
    codesToCommands.put(10, " ");
    commandsToCodes.put(" ", 10);

    codesToCommands.put(11, "#");
    commandsToCodes.put("#", 11);

    codesToCommands.put(12, "Go");
    commandsToCodes.put("Go", 12);

    codesToCommands.put(13, "End");
    commandsToCodes.put("End", 13);  
    
//    System.err.println(codesToCommands);
//    System.exit(1);
  }
  
  public Integer getCode(String command) {
    return commandsToCodes.get(command);
  }
  
  public String getCommand(int code) {
    return codesToCommands.get(code);
  }
}
