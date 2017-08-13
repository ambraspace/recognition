package com.ambraspace.recognition.opencv;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import java.io.File;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FaceDetect {
	
	private static final Logger log = LoggerFactory.getLogger(FaceDetect.class);
	
	
	private static class FaceCoordinates {
		
		private final int x, y, w, h;
		
		public FaceCoordinates(int x, int y, int width, int height) {

			this.x = x;
			this.y = y;
			this.w = width;
			this.h = height;
			
		}
		
		public int getX() {
			return x;
		}
		
		public int getY() {
			return y;
		}
		
		public int getWidth() {
			return w;
		}
		
		public int getHeight() {
			return h;
		}
		
		
	}
	

	public static void main(String[] args) {
		
		File currentFile = null;
		FaceCoordinates[] faces = null;
		for (int i=0; i<args.length; i++) {
			
			currentFile = new File(args[i]);
			if (currentFile.exists()) {
				
				faces = getFaceCoordinates(currentFile);
				if (faces == null || faces.length==0)
					continue;
				FaceCoordinates centeredFace = faces[0];
				for (int j=0; j<faces.length; j++) {
					if (deviationFromCenter(centeredFace)>deviationFromCenter(faces[j])) {
						centeredFace=faces[j];
					}
				}
				System.out.printf("%s\t%d\t%d\t%d\t%d\n",
						currentFile.getName(),
						centeredFace.getX(),
						centeredFace.getY(),
						centeredFace.getWidth(),
						centeredFace.getHeight());
			} else {
				log.error("File " + currentFile.getAbsolutePath() + " doesn't exist!");
			}

			
		}
		
	}
	
	
	private static double deviationFromCenter(FaceCoordinates face) {
		double tmp1, tmp2;
		tmp1 = 125-(face.getX()+face.getWidth()/2);
		tmp2 = 125-(face.getY()+face.getHeight()/2);
		return Math.sqrt(Math.pow(tmp1, 2.0)+Math.pow(tmp2, 2.0));
	}
	
	private static FaceCoordinates[] getFaceCoordinates(File file) {
		
		String facesFile = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
		String eyesFile = "/usr/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
		
		CascadeClassifier faces = new CascadeClassifier();
		CascadeClassifier eyes = new CascadeClassifier();
		
		faces.load(facesFile);
		eyes.load(eyesFile);
		
		Mat image = imread(file.getAbsolutePath(), CV_LOAD_IMAGE_COLOR);
		if (image.empty()) {
			log.error("Can not read " + file.getName());
			faces.close();
			eyes.close();
			return null;
		}
		
		RectVector results = new RectVector();
		
		faces.detectMultiScale(image, results);
		
		FaceCoordinates[] retVal = new FaceCoordinates[(int)results.size()];
		
		for (int k=0; k<results.size(); k++) {
			retVal[k] = new FaceCoordinates(
					results.get(k).x(),
					results.get(k).y(),
					results.get(k).width(),
					results.get(k).height());
		}
		
		faces.close();
		eyes.close();
				
		return retVal;

	}
	
}
