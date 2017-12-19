package org.jol.core;

import java.util.HashMap;
import java.util.Map;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;

public class DSL {
  private static final Map<Integer, String> codesToCommands = new HashMap<Integer, String>();
  private static final Map<String, Integer> commandsToCodes = new HashMap<String, Integer>();
  private ScriptEngine engine;

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

	codesToCommands.put(52, "K");
    commandsToCodes.put("K", 52);

    codesToCommands.put(53, "L");
    commandsToCodes.put("L", 53);

    codesToCommands.put(54, "M");
    commandsToCodes.put("M", 54);

    codesToCommands.put(55, "N");
    commandsToCodes.put("N", 55);

    codesToCommands.put(56, "print('");
    commandsToCodes.put("print('", 56);	

    codesToCommands.put(57, ";");
    commandsToCodes.put(";", 57);	

    codesToCommands.put(58, "')");
    commandsToCodes.put("')", 58);	

    // create a script engine manager
    ScriptEngineManager factory = new ScriptEngineManager();
    // create a JavaScript engine
    engine = factory.getEngineByName("JavaScript");
  }
  
  public Integer getCode(String command) {
    return commandsToCodes.get(command);
  }
  
  public String getCommand(int code) {
    return codesToCommands.get(code);
  }
  
  public void act(String comm) throws ScriptException {
    // evaluate JavaScript code from String
    engine.eval(comm);
  }
}
