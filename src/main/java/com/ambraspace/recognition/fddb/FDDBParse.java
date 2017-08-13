package com.ambraspace.recognition.fddb;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FDDBParse {
	
	private static final Logger log = LoggerFactory.getLogger(FDDBParse.class);
	
	public static void main(String[] args) {
		
		String workDir = "/home/ambra/Deep Learning/FDDB";
		
		File currentFile = null;
		for (int i=1; i<=1; i++) {

			currentFile = new File(workDir +
					"/FDDB-folds/FDDB-fold-" +
					String.format("%02d", i) +
					"-ellipseList.txt");
			
			try (BufferedReader fin = new BufferedReader(new InputStreamReader(new FileInputStream(currentFile)))) {
				
				String line = null;
				String file = null;
				Integer numOfFaces = 0;
				while ((line = fin.readLine()) != null) {
					if (line.contains("/")) {
						file = line;
						line = fin.readLine();
						numOfFaces = Integer.parseInt(line);
						for (int j=0; j<numOfFaces; j++) {
							line = fin.readLine();
							System.out.printf("%s\t%s\n", file, line.replace(" ", "\t"));
						}
					}
				}

			} catch (FileNotFoundException e) {
				log.error("File " + currentFile.getAbsolutePath() + " doesn't exist!");
			} catch (IOException e) {
				log.error("Can not read " + currentFile.getAbsolutePath() + "!");
			}

			

		}
		
	}

}
