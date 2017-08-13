package com.ambraspace.recognition.opencv;

import java.io.File;
import java.util.Random;


import org.bytedeco.javacpp.opencv_core.Mat;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.SimpleLayout;
import org.apache.log4j.spi.Filter;
import org.apache.log4j.spi.LoggingEvent;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_core.*;


public class NegativeSampleGenerator {
	
	private static final Logger log = LoggerFactory.getLogger(NegativeSampleGenerator.class);
	
	public static void main(String[] args) {
	
		File destinationFolder = new File("/home/ambra/Deep Learning/LFW/detection-00");
		
		File currentFile = null;

		for (int i=0; i<args.length; i++) {
			
			currentFile = new File(args[i]);
			Mat image = null;
			int imageWidth, imageHeight;
			String imageName = null;
			
			if (currentFile.exists()) {
				
				log.info(currentFile.getAbsolutePath());
				
				image = imread(currentFile.getAbsolutePath());
				if (image.empty()) {
					log.error("Can not read " + currentFile.getName());
					continue;
				}
				imageWidth = image.cols();
				imageHeight = image.rows();
				if (imageWidth<51) continue;
				if (imageHeight<51) continue;
				
				imageName = currentFile.getName().substring(0, currentFile.getName().lastIndexOf("."));
				
				Mat subImage = null;
				Mat mean = new Mat(), deviation = new Mat();
				int iColRange, iRowRange;
				iColRange = (imageWidth-51)/51;
				iRowRange = (imageHeight-51)/51;
				DoubleIndexer di;
				for (int row=0; row<=iRowRange; row++) {
					for (int col=0; col<=iColRange; col++) {
						subImage = image.colRange(col*51, (col+1)*51).rowRange(row*51, (row+1)*51);
						meanStdDev(subImage, mean, deviation);
						di = deviation.createIndexer();
						if (stDevOK(di)) {
							File outFile = new File(destinationFolder.getAbsolutePath() +
									File.separator +
									imageName + "-" + row + "-" + col +".png");
							if (!outFile.exists()) {
								imwrite(
										outFile.getAbsolutePath(),
										subImage);
							}

						}
						mean.release();
						deviation.release();
					}
				}

				image.release();

			} else {
				log.error("File " + currentFile.getAbsolutePath() + " doesn't exist!");
			}

			
		}

		
		
	}
	
	
	public static boolean stDevOK(DoubleIndexer di) {
		
		if (di.rows()==1) {
			if (di.get(0,0)>=25.0d) return true;
		} else if (di.rows()==3) {
			if (di.get(0,0)>=25.0d) return true;
			if (di.get(0,1)>=25.0d) return true;
			if (di.get(0,2)>=25.0d) return true;
		}
		
		return false;
	}

}
